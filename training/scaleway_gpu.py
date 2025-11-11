"""Scaleway GPU instance helper.

Note: Created with ChatGPT-5 agent.

Features:
	- Idempotent start/stop
	- Status polling & convenience start_and_wait/stop_and_wait
	- Simple exponential backoff retries for transient HTTP/network errors

Environment Variables:
	SCW_SECRET_KEY (required)
	SCW_SERVER_ID (required)
	SCW_ZONE       (optional, default: fr-par-1)

Minimal example:
	from training.scaleway_gpu import ScalewayGPU
	gpu = ScalewayGPU()
	print(gpu.status())
	gpu.start_and_wait()
	# ... work ...
	gpu.stop_and_wait()
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

__all__ = ["ScalewayGPU"]

_DEFAULT_ZONE = "fr-par-1"

class ScalewayGPU:
	def __init__(
		self,
		*,
		secret_key: Optional[str] = None,
		zone: Optional[str] = None,
		server_id: Optional[str] = None,
		max_retries: int = 3,
		backoff_base: float = 1.0,
		verbose: bool = False,
		running_timeout: int = 300,
		stopping_timeout: int = 180,
		poll_interval: float = 5.0,
	) -> None:
		"""Create a client.

		Parameters
		----------
		max_retries : int
			Number of attempts for transient errors (>=1).
		backoff_base : float
			Base seconds for exponential backoff (delay = base * 2^(attempt-1)).
		running_timeout : int
			Timeout for running state (default: 300 seconds).
		stopping_timeout : int
			Timeout for stopping state (default: 180 seconds).
		poll_interval : float
			Interval for polling state (default: 5 seconds).
		"""
		self.secret_key = secret_key or os.getenv("SCW_SECRET_KEY")
		self.zone = zone or os.getenv("SCW_ZONE") or _DEFAULT_ZONE
		self.server_id = server_id or os.getenv("SCW_SERVER_ID")
		self._base = "https://api.scaleway.com"
		self.max_retries = max(1, max_retries)
		self.backoff_base = max(0.1, backoff_base)
		self.verbose = verbose
		# Wait/timeout configuration
		self.running_timeout = int(running_timeout)
		self.stopping_timeout = int(stopping_timeout)
		self.poll_interval = float(poll_interval)

		if not self.secret_key:
			raise ValueError("SCW_SECRET_KEY is required")
		if not self.server_id:
			raise ValueError("SCW_SERVER_ID is required")

	def __enter__(self) -> ScalewayGPU:
		"""Context manager entry: returns self."""
		return self

	def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
		"""Context manager exit: stops the instance."""
		self._log("Exiting context; stopping instance...")
		self.stop()
	
	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------
	def connect(self) -> str:
		"""Validate access by returning current instance status."""
		return self.status()

	def start(self, wait_if_stopping: bool = True) -> Dict[str, Any]:
		"""Power on the instance (async).

		Idempotent when already running/starting. If the instance is currently
		stopping, optionally wait until it is fully stopped before starting
		(default behavior).
		"""
		state = self.status()
		if state in ("running", "starting"):
			return self._handle_already_running(state)

		if state == "stopping" and wait_if_stopping:
			self._log("Instance is currently stopping; waiting for 'stopped' before starting...")
			waited = self.wait_for("stopped", timeout=self.stopping_timeout, poll_interval=self.poll_interval)
			if not waited:
				# Do not fail hard; report current state so caller can decide to retry
				self._log("Instance still stopping; start deferred.")
				return {"skipped": True, "state": self.status(), "message": "Instance still stopping; start deferred"}

		try:
			self._log("Sending poweron action...")
			return self._action("poweron")
		except urllib.error.HTTPError as e:  # pragma: no cover - depends on external API
			if e.code in (400, 409):  # conflict / invalid transition
				cur = self.status()
				if cur in ("running", "starting"):
					return self._handle_already_running(cur)
				if cur == "stopping" and wait_if_stopping:
					# One more wait attempt, then retry once
					self._log("Instance is currently stopping; waiting again for 'stopped' before retrying poweron...")
					if self.wait_for("stopped", timeout=self.stopping_timeout, poll_interval=self.poll_interval):
						self._log("Stopped reached; retrying poweron...")
						return self._action("poweron")
					self._log("Instance still stopping; start deferred.")
					return {"skipped": True, "state": self.status(), "message": "Instance still stopping; start deferred"}
			raise

	def stop(self) -> Dict[str, Any]:
		"""Power off the instance (async). Idempotent if already stopped/stopping."""
		state = self.status()
		if state in ("stopped", "stopping"):
			return self._handle_already_stopped(state)
		try:
			self._log("Sending poweroff action...")
			return self._action("poweroff")
		except urllib.error.HTTPError as e:  # pragma: no cover - depends on external API
			if e.code in (400, 409):
				cur = self.status()
				if cur in ("stopped", "stopping"):
					return self._handle_already_stopped(cur)
			raise

	def status(self) -> str:
		"""Return the instance state (e.g., running, stopped)."""
		path = f"/instance/v1/zones/{self.zone}/servers/{self.server_id}"
		data = self._request("GET", path)
		server = data.get("server", {}) if isinstance(data, dict) else {}
		return server.get("state", "unknown")

	def wait_for(self, target_state: str, timeout: Optional[int] = None, poll_interval: Optional[float] = None) -> bool:
		"""Poll until the instance reaches target_state or timeout expires.

		Returns True if reached, False otherwise.
		"""
		if timeout is None:
			# Default timeout based on typical transitions
			timeout = self.running_timeout if target_state == "running" else self.stopping_timeout
		if poll_interval is None:
			poll_interval = self.poll_interval

		self._log(f"Waiting for state '{target_state}' (timeout={timeout}s, every {poll_interval}s)...")
		end = time.time() + timeout
		while time.time() < end:
			if self.status() == target_state:
				self._log(f"Reached state '{target_state}'.")
				return True
			time.sleep(poll_interval)
		self._log(f"Timeout waiting for state '{target_state}'.")
		return False

	def start_and_wait(self) -> bool:
		"""Start instance then wait for 'running'. Returns success bool."""
		self.start()
		return self.wait_for("running")

	def stop_and_wait(self) -> bool:
		"""Stop instance then wait for 'stopped'. Returns success bool."""
		self.stop()
		return self.wait_for("stopped")

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------
	def _handle_already_running(self, state: str) -> Dict[str, Any]:
		msg = "starting" if state == "starting" else "running"
		self._log(f"Instance is currently {msg}; start skipped.")
		return {"skipped": True, "state": state, "message": "Instance already running or starting"}

	def _handle_already_stopped(self, state: str) -> Dict[str, Any]:
		msg = "stopping" if state == "stopping" else "stopped"
		self._log(f"Instance is currently {msg}; stop skipped.")
		return {"skipped": True, "state": state, "message": "Instance already stopped or stopping"}

	def _action(self, action: str) -> Dict[str, Any]:
		path = f"/instance/v1/zones/{self.zone}/servers/{self.server_id}/action"
		return self._request("POST", path, {"action": action})

	def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""HTTP request with simple exponential backoff on transient failures."""
		url = f"{self._base}{path}"
		headers = {
			"X-Auth-Token": self.secret_key,
			"Accept": "application/json",
			"Content-Type": "application/json",
		}
		data = json.dumps(body).encode() if body is not None else None
		attempt = 0
		while True:
			attempt += 1
			req = urllib.request.Request(url, data=data, headers=headers, method=method)
			try:
				with urllib.request.urlopen(req, timeout=30) as resp:  # pragma: no cover - network
					raw = resp.read()
					return json.loads(raw.decode()) if raw else {}
			except urllib.error.HTTPError as e:  # pragma: no cover - network
				# Retry on 5xx and 429 (rate limit). Otherwise raise immediately.
				if e.code in (429,) or 500 <= e.code < 600:
					if attempt < self.max_retries:
						self._log(f"HTTP error {e.code}, retrying (attempt {attempt}/{self.max_retries})...")
						self._backoff_sleep(attempt)
						continue
					else:
						raise
				else:
					raise
			except urllib.error.URLError:  # pragma: no cover - network
				if attempt < self.max_retries:
					self._backoff_sleep(attempt)
					continue
				else:
					raise

	def _backoff_sleep(self, attempt: int) -> None:
		# attempt starts at 1; delay escalates: base * 2^(attempt-1)
		delay = self.backoff_base * (2 ** (attempt - 1))
		time.sleep(delay)

	def __repr__(self) -> str:  # Helpful for debugging
		return (
			f"ScalewayGPU(server_id={self.server_id!r}, zone={self.zone!r}, "
			f"max_retries={self.max_retries}, backoff_base={self.backoff_base})"
		)

	# ------------------------------------------------------------------
	# Logging
	# ------------------------------------------------------------------
	def _log(self, message: str) -> None:
		if self.verbose:
			print(f"[ScalewayGPU] {message}")

