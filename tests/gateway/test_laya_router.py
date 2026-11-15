import io
import json
import unittest

from gateway.laya_router import DEFAULT_LANE, reasoning_for_lane, route_hermes_request


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


if __name__ == "__main__":
    unittest.main()
