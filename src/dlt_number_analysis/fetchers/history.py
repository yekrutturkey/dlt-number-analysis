"""Network-only fetchers for independently sourced DLT history snapshots."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dlt_number_analysis.data.source_import import SourceSnapshot

SPORTTERY_ENDPOINT = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
FIVE_HUNDRED_ENDPOINT = "https://datachart.500.com/dlt/history/newinc/history.php"


def _read_url(
    url: str,
    *,
    timeout_seconds: float,
    referer: str | None = None,
) -> bytes:
    headers = {
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/138.0 Safari/537.36"
        ),
    }
    if referer is not None:
        headers["Referer"] = referer
    curl = shutil.which("curl")
    if sys.platform == "win32" and curl is not None:
        arguments = [
            curl,
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--ssl-no-revoke",
            "--max-time",
            str(timeout_seconds),
            "--user-agent",
            headers["User-Agent"],
        ]
        if referer is not None:
            arguments.extend(["--referer", referer])
        arguments.append(url)
        return subprocess.run(
            arguments,
            check=True,
            capture_output=True,
        ).stdout
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def _read_url_with_retries(
    url: str,
    *,
    timeout_seconds: float,
    referer: str | None = None,
    attempts: int = 3,
) -> bytes:
    """Retry transient transport failures without accepting invalid response bodies."""
    last_error: OSError | subprocess.SubprocessError | None = None
    for attempt in range(attempts):
        try:
            return _read_url(url, timeout_seconds=timeout_seconds, referer=referer)
        except (OSError, subprocess.SubprocessError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


class SportteryHistoryFetcher:
    """Fetch official pages while preserving every exact raw response body."""

    def __init__(
        self,
        *,
        page_size: int = 30,
        timeout_seconds: float = 30.0,
        request_interval_seconds: float = 0.2,
    ) -> None:
        if page_size < 1:
            raise ValueError("page_size must be positive")
        if timeout_seconds <= 0 or request_interval_seconds < 0:
            raise ValueError("timeouts must be positive and request interval non-negative")
        self.page_size = page_size
        self.timeout_seconds = timeout_seconds
        self.request_interval_seconds = request_interval_seconds

    def _page_url(self, page_number: int) -> str:
        query = urlencode(
            {
                "gameNo": "85",
                "provinceId": "0",
                "pageSize": self.page_size,
                "isVerify": "1",
                "pageNo": page_number,
            }
        )
        return f"{SPORTTERY_ENDPOINT}?{query}"

    def fetch(self, *, fetched_at: datetime | None = None) -> SourceSnapshot:
        """Fetch all advertised pages into a hash-checked raw-body bundle."""
        pages: list[dict[str, object]] = []
        page_number = 1
        advertised_pages: int | None = None
        while advertised_pages is None or page_number <= advertised_pages:
            url = self._page_url(page_number)
            body = _read_url_with_retries(
                url,
                timeout_seconds=self.timeout_seconds,
                referer="https://m.lottery.gov.cn/zst/dlt/",
            )
            payload = json.loads(body.decode("utf-8-sig"))
            value = payload.get("value") if isinstance(payload, dict) else None
            if not isinstance(value, dict) or not isinstance(value.get("pages"), int):
                raise ValueError("official source did not return a valid history page")
            current_pages = int(value["pages"])
            if advertised_pages is None:
                advertised_pages = current_pages
            elif current_pages != advertised_pages:
                raise ValueError("official source page count changed during fetch")
            pages.append(
                {
                    "page_number": page_number,
                    "url": url,
                    "content_hash": hashlib.sha256(body).hexdigest(),
                    "body_base64": base64.b64encode(body).decode("ascii"),
                }
            )
            page_number += 1
            if advertised_pages is not None and page_number <= advertised_pages:
                time.sleep(self.request_interval_seconds)
        bundle = json.dumps(
            {
                "schema_version": "sporttery-raw-page-bundle-v1",
                "page_size": self.page_size,
                "pages": pages,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return SourceSnapshot.from_content(
            source_name="china_sports_lottery_official",
            source_url=SPORTTERY_ENDPOINT,
            source_format="sporttery_json_pages",
            content=bundle,
            fetched_at=fetched_at,
        )


class FiveHundredHistoryFetcher:
    """Fetch one independent 500.com history table response."""

    def __init__(
        self,
        *,
        start_issue: str = "07001",
        end_issue: str = "99999",
        timeout_seconds: float = 60.0,
    ) -> None:
        if not start_issue.isdigit() or not end_issue.isdigit():
            raise ValueError("issue bounds must contain digits only")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.start_issue = start_issue
        self.end_issue = end_issue
        self.timeout_seconds = timeout_seconds

    @property
    def url(self) -> str:
        query = urlencode({"start": self.start_issue, "end": self.end_issue})
        return f"{FIVE_HUNDRED_ENDPOINT}?{query}"

    def fetch(self, *, fetched_at: datetime | None = None) -> SourceSnapshot:
        """Fetch the independent raw HTML response without normalizing it."""
        return SourceSnapshot.from_content(
            source_name="500_com",
            source_url=self.url,
            source_format="five_hundred_html",
            content=_read_url_with_retries(self.url, timeout_seconds=self.timeout_seconds),
            fetched_at=fetched_at,
        )
