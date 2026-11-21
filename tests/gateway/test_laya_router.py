import io
import json
import unittest

from gateway.laya_router import DEFAULT_LANE, reasoning_for_lane, route_hermes_request, runtime_for_lane


class Response:
    def __init__(self, payload, status=200):
        self.status = status
        self._body = io.BytesIO(json.dumps(payload).encode())

    def read(self):
        return self._body.read()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class LayaRouterTests(unittest.TestCase):
    def test_fast_lane_uses_bounded_local_lfm_runtime(self):
        calls = []

        def resolve(provider, target_model=None):
            calls.append((provider, target_model))
            return {
                "provider": "custom",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_mode": "chat_completions",
            }

        model, runtime = runtime_for_lane(
            "FAST", "gpt-5.6-sol", {"provider": "openai-codex"}, resolver=resolve,
        )
        self.assertEqual(model, "lfm3:1.7b")
        self.assertEqual(runtime["provider"], "custom")
        self.assertEqual(calls, [("ollama", "lfm3:1.7b")])

    def test_non_fast_and_resolution_failure_preserve_existing_runtime(self):
        original = {"provider": "openai-codex"}
        for lane in ("FULL", "HEAVY", "INVALID"):
            model, runtime = runtime_for_lane(lane, "gpt-5.6-sol", original)
            self.assertEqual(model, "gpt-5.6-sol")
            self.assertIs(runtime, original)

        def broken(*_args, **_kwargs):
            raise RuntimeError("offline")

        model, runtime = runtime_for_lane(
            "FAST", "gpt-5.6-sol", original, resolver=broken,
        )
        self.assertEqual(model, "gpt-5.6-sol")
        self.assertIs(runtime, original)

    def test_reuses_injected_http_client(self):
        class HttpResponse:
            status_code = 200

            def json(self):
                return {"hermes": {"class": "FAST"}}

        class Client:
            def __init__(self):
                self.calls = []

            def post(self, url, *, json, timeout):
                self.calls.append((url, json, timeout))
                return HttpResponse()

        client = Client()
        self.assertEqual(route_hermes_request("one", client=client), "FAST")
        self.assertEqual(route_hermes_request("two", client=client), "FAST")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0][1], {"text": "one", "target": "hermes"})
        self.assertEqual(client.calls[1][1], {"text": "two", "target": "hermes"})

    def test_routes_supported_lanes_and_posts_hermes_target(self):
        for lane in ("FAST", "FULL", "HEAVY"):
            seen = {}

            def opener(request, timeout):
                seen["payload"] = json.loads(request.data)
                seen["timeout"] = timeout
                return Response({"hermes": {"class": lane}})

            self.assertEqual(route_hermes_request("hello", opener=opener), lane)
            self.assertEqual(seen["payload"], {"text": "hello", "target": "hermes"})
            self.assertEqual(seen["timeout"], 0.75)

    def test_failures_and_malformed_outputs_fail_upward_to_full(self):
        cases = [
            lambda *_args, **_kwargs: Response({}, 200),
            lambda *_args, **_kwargs: Response({"hermes": {"class": "SIMPLE"}}, 200),
            lambda *_args, **_kwargs: Response({}, 503),
        ]
        for opener in cases:
            self.assertEqual(route_hermes_request("hello", opener=opener), DEFAULT_LANE)

        def broken(*_args, **_kwargs):
            raise TimeoutError("late")

        self.assertEqual(route_hermes_request("hello", opener=broken), DEFAULT_LANE)

    def test_lane_reasoning_is_request_scoped(self):
        existing = {"effort": "medium"}
        self.assertIs(reasoning_for_lane("FULL", existing), existing)
        fast = reasoning_for_lane("FAST", existing)
        heavy = reasoning_for_lane("HEAVY", existing)
        self.assertIsNotNone(fast)
        self.assertIsNotNone(heavy)
        assert fast is not None and heavy is not None
        self.assertEqual(fast["effort"], "low")
        self.assertEqual(heavy["effort"], "high")
        self.assertIs(reasoning_for_lane("INVALID", existing), existing)


if __name__ == "__main__":
    unittest.main()
