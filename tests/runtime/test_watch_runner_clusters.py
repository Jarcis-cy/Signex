from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.runtime.watch_runner import (
    NEED_CLUSTER_SOURCE,
    NeedCluster,
    PreprocessRunSummary,
    _cluster_need_candidates,
    _persist_need_clusters,
    _render_report,
    _select_need_candidates,
)
from src.store.database import Database


class WatchRunnerClusterTests(unittest.TestCase):
    def test_select_need_candidates_excludes_need_cluster_source(self) -> None:
        items = [
            {
                "id": 1,
                "source": NEED_CLUSTER_SOURCE,
                "preprocessStatus": "valid",
                "isRelevant": 1,
                "normalizedTitle": "Cluster card",
                "normalizedContent": "Already processed cluster",
            },
            {
                "id": 2,
                "source": "reddit",
                "preprocessStatus": "valid",
                "isRelevant": 1,
                "normalizedTitle": "Need signal",
                "normalizedContent": "User asks for feature",
            },
        ]

        selected = _select_need_candidates(items)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["id"], 2)

    def test_select_need_candidates_uses_source_diverse_round_robin(self) -> None:
        items = [
            {
                "id": 10,
                "source": "web_search",
                "preprocessStatus": "valid",
                "isRelevant": 1,
                "normalizedTitle": "Need 10",
                "normalizedContent": "Need content 10",
                "published_at": "2026-03-05T10:05:00+00:00",
            },
            {
                "id": 11,
                "source": "web_search",
                "preprocessStatus": "valid",
                "isRelevant": 1,
                "normalizedTitle": "Need 11",
                "normalizedContent": "Need content 11",
                "published_at": "2026-03-05T10:04:00+00:00",
            },
            {
                "id": 12,
                "source": "web_search",
                "preprocessStatus": "valid",
                "isRelevant": 1,
                "normalizedTitle": "Need 12",
                "normalizedContent": "Need content 12",
                "published_at": "2026-03-05T10:03:00+00:00",
            },
            {
                "id": 20,
                "source": "hacker_news",
                "preprocessStatus": "valid",
                "isRelevant": 1,
                "normalizedTitle": "Need 20",
                "normalizedContent": "Need content 20",
                "published_at": "2026-03-05T10:02:00+00:00",
            },
            {
                "id": 30,
                "source": "product_hunt",
                "preprocessStatus": "valid",
                "isRelevant": 1,
                "normalizedTitle": "Need 30",
                "normalizedContent": "Need content 30",
                "published_at": "2026-03-05T10:01:00+00:00",
            },
        ]

        with patch.dict(os.environ, {"SIGNEX_MAX_NEED_ANALYSIS_ITEMS": "4"}, clear=False):
            selected = _select_need_candidates(items)

        self.assertEqual([item["id"] for item in selected], [10, 20, 30, 11])

    def test_render_report_includes_candidate_funnel(self) -> None:
        items = [
            {
                "id": 10,
                "source": "web_search",
                "title": "Need a faster export",
                "normalizedTitle": "Need a faster export",
                "normalizedContent": "Power users want async export with progress view",
                "preprocessStatus": "valid",
                "isRelevant": 1,
                "published_at": "2026-03-05T10:05:00+00:00",
            },
            {
                "id": 11,
                "source": "hacker_news",
                "title": "Need fewer timeouts",
                "normalizedTitle": "Need fewer timeouts",
                "normalizedContent": "Users report recurring timeout issues",
                "preprocessStatus": "valid",
                "isRelevant": 1,
                "published_at": "2026-03-05T10:04:00+00:00",
            },
            {
                "id": 12,
                "source": "web_search",
                "title": "Noise result",
                "normalizedTitle": "Noise result",
                "normalizedContent": "Mostly irrelevant source",
                "preprocessStatus": "irrelevant",
                "isRelevant": 0,
                "published_at": "2026-03-05T10:03:00+00:00",
            },
        ]
        report = _render_report(
            "demo",
            "deep_insight",
            items,
            [],
            items[:2],
            PreprocessRunSummary(total_targets=3, llm_batches=1, success_count=3, failure_count=0),
            "en",
            [],
            datetime(2026, 3, 5, 10, 30, tzinfo=timezone.utc),
        )

        self.assertIn("Relevant need candidates available: 2", report)
        self.assertIn("Candidates passed to clustering: 2", report)
        self.assertIn("Relevant candidate sources: hacker_news:1, web_search:1", report)
        self.assertIn("Cluster input sources: hacker_news:1, web_search:1", report)

    def test_cluster_need_candidates_normalizes_output(self) -> None:
        candidates = [
            {
                "id": 10,
                "source": "reddit",
                "normalizedTitle": "Need a faster export",
                "normalizedContent": "Power users want async export with progress view",
                "url": "https://example.com/a",
            },
            {
                "id": 11,
                "source": "v2ex",
                "normalizedTitle": "CSV export timeout",
                "normalizedContent": "Export jobs timeout when data set is large",
                "url": "https://example.com/b",
            },
        ]

        with patch("src.runtime.watch_runner._run_claude_json") as mocked:
            mocked.return_value = (
                {
                    "clusters": [
                        {
                            "clusterId": "Export Reliability",
                            "needTitle": "导出可靠性提升",
                            "needSummary": "用户希望大数据量导出可稳定完成并有状态反馈。",
                            "painPoint": "当前导出容易超时失败，重试成本高。",
                            "targetUser": "有批量数据导出需求的运营和分析用户",
                            "suggestedDirection": "引入异步导出队列与进度通知",
                            "whyNow": "近期讨论密度明显提升",
                            "confidence": 83,
                            "evidenceItemIds": [10, 11],
                            "evidenceUrls": ["https://example.com/a"],
                            "tags": ["export", "reliability"],
                        }
                    ]
                },
                "",
            )

            clusters = _cluster_need_candidates(
                root_dir=Path("/tmp"),
                watch_name="demo",
                intent_text="track export pain",
                memory_text="",
                language_code="zh",
                candidates=candidates,
            )

        self.assertEqual(len(clusters), 1)
        cluster = clusters[0]
        self.assertEqual(cluster.cluster_id, "export-reliability")
        self.assertEqual(cluster.evidence_item_ids, [10, 11])
        self.assertIn("https://example.com/a", cluster.evidence_urls)
        self.assertEqual(cluster.confidence, 83)

    def test_cluster_need_candidates_falls_back_when_claude_fails(self) -> None:
        candidates = [
            {
                "id": 10,
                "source": "reddit",
                "normalizedTitle": "Need a faster export",
                "normalizedContent": "Power users want async export with progress view",
            }
        ]

        with patch("src.runtime.watch_runner._run_claude_json") as mocked:
            mocked.side_effect = RuntimeError("claude failed")
            clusters = _cluster_need_candidates(
                root_dir=Path("/tmp"),
                watch_name="demo",
                intent_text="track export pain",
                memory_text="",
                language_code="en",
                candidates=candidates,
            )

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].evidence_item_ids, [10])
        self.assertTrue(clusters[0].title)

    def test_persist_need_clusters_upserts_cluster_cards(self) -> None:
        now_iso = datetime(2026, 3, 5, 10, 0, tzinfo=timezone.utc).isoformat()
        candidates = [
            {
                "id": 21,
                "source": "reddit",
                "url": "https://example.com/c",
                "published_at": now_iso,
            }
        ]
        clusters = [
            NeedCluster(
                cluster_id="export-reliability",
                title="导出可靠性提升",
                summary="大数据量导出需要异步队列和进度反馈。",
                pain_point="超时失败频发",
                target_user="数据运营",
                suggested_direction="异步导出 + 回调通知",
                why_now="近期重复出现",
                confidence=78,
                evidence_item_ids=[21],
                evidence_urls=["https://example.com/c"],
                tags=["export"],
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            persisted = _persist_need_clusters(
                root_dir=Path(tmp),
                watch_name="demo-watch",
                language_code="zh",
                preprocess_run_id="watch-llm-2026-03-05T10:00:00+00:00",
                preprocessed_at=now_iso,
                clusters=clusters,
                candidates=candidates,
            )

            self.assertEqual(len(persisted), 1)
            self.assertGreater(persisted[0].item_id, 0)

            db = Database(str(Path(tmp) / "data/signex.db"))
            db.init()
            try:
                rows = db.get_items(source=NEED_CLUSTER_SOURCE)
            finally:
                db.close()

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["preprocessStatus"], "valid")
        self.assertEqual(row["isRelevant"], 1)
        self.assertEqual(row["normalizedTitle"], "导出可靠性提升")
        self.assertEqual(row["source"], NEED_CLUSTER_SOURCE)


if __name__ == "__main__":
    unittest.main()
