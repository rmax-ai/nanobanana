import base64
import json
from pathlib import Path

import pytest

import nanobanana as nb  # type: ignore[import]


class TestPromptAssembly:
    def test_expand_negative_splits_and_bullets(self):
        result = nb.expand_negative("no text, no people; blur")
        assert result == "- no text\n- no people\n- blur"

    def test_expand_negative_empty(self):
        assert nb.expand_negative("") == ""
        assert nb.expand_negative("   ") == ""

    def test_annotate_references_empty(self):
        assert nb.annotate_references(()) == ""

    def test_annotate_references_roles(self):
        refs = (
            nb.ReferenceImage(
                path=Path("a.png"), role="subject", mime_type="image/png", sha256="x"
            ),
            nb.ReferenceImage(
                path=Path("b.png"), role="style", mime_type="image/png", sha256="y"
            ),
        )
        result = nb.annotate_references(refs)
        assert "Reference 1 (subject)" in result
        assert "Reference 2 (style)" in result

    def test_build_normalized_prompt_has_sections(self):
        req = nb.ImageRequest(
            command="generate",
            prompt="A test",
            model="auto",
            aspect_ratio="16:9",
            image_size="2K",
        )
        prompt = nb.build_normalized_prompt(req, negative="no text, no people")
        assert "TASK:" in prompt
        assert "REQUIREMENTS:" in prompt
        assert "AVOID:" in prompt
        assert "OUTPUT INTENT:" in prompt
        assert "no text" in prompt
        assert "16:9" in prompt

    def test_build_normalized_prompt_with_preset_prefix(self):
        req = nb.ImageRequest(
            command="generate",
            prompt="A test",
            model="auto",
        )
        preset = {"prompt_prefix": "Create a clean icon"}
        prompt = nb.build_normalized_prompt(req, preset=preset)
        assert "CONTEXT:" in prompt
        assert "Create a clean icon" in prompt
        assert "A test" in prompt

    def test_build_normalized_prompt_preserve_for_edit(self):
        req = nb.ImageRequest(
            command="edit",
            prompt="Change color",
            model="auto",
        )
        prompt = nb.build_normalized_prompt(req)
        assert "PRESERVE:" in prompt


class TestInteractionInput:
    def test_build_interaction_input_basic(self):
        req = nb.ImageRequest(
            command="generate",
            prompt="A test",
            model="auto",
            aspect_ratio="16:9",
            image_size="2K",
        )
        decision = nb.ModelDecision(
            requested="auto",
            resolved="gemini-3.1-flash-image",
            selection_reason="default",
        )
        data = nb.build_interaction_input(req, "prompt text", decision)
        assert data["model"] == "gemini-3.1-flash-image"
        assert data["response_format"] == {"type": "image/png"}
        assert data["contents"] == [
            {"role": "user", "parts": [{"text": "prompt text"}]}
        ]

    def test_build_interaction_input_text_output(self):
        req = nb.ImageRequest(
            command="generate",
            prompt="A test",
            model="auto",
            text_output=True,
        )
        decision = nb.ModelDecision(
            requested="auto",
            resolved="gemini-3.1-flash-image",
            selection_reason="default",
        )
        data = nb.build_interaction_input(req, "prompt text", decision)
        assert data["response_format"] == [
            {"type": "text"},
            {"type": "image/png"},
        ]

    def test_build_interaction_input_grounding(self):
        req = nb.ImageRequest(
            command="grounded",
            prompt="News",
            model="auto",
            grounding="web",
        )
        decision = nb.ModelDecision(
            requested="auto",
            resolved="gemini-3.1-flash-image",
            selection_reason="default",
        )
        data = nb.build_interaction_input(req, "prompt text", decision)
        assert data["tools"] == [{"type": "google_search"}]

    def test_build_interaction_input_with_references(self, tmp_path):
        img_path = tmp_path / "ref.png"
        img_path.write_bytes(b"\x89PNG\x00\x00")
        ref = nb.load_reference(img_path, role="subject")
        req = nb.ImageRequest(
            command="compose",
            prompt="Combine",
            model="auto",
            references=(ref,),
        )
        decision = nb.ModelDecision(
            requested="auto",
            resolved="gemini-3.1-flash-image",
            selection_reason="default",
        )
        data = nb.build_interaction_input(req, "prompt text", decision)
        parts = data["contents"][0]["parts"]
        assert parts[0] == {"text": "prompt text"}
        assert parts[1]["inline_data"]["mime_type"] == "image/png"
        assert base64.b64decode(parts[1]["inline_data"]["data"]) == b"\x89PNG\x00\x00"

    def test_build_interaction_input_thinking_and_seed(self):
        req = nb.ImageRequest(
            command="generate",
            prompt="A test",
            model="auto",
            thinking_level="high",
            seed="42",
        )
        decision = nb.ModelDecision(
            requested="auto",
            resolved="gemini-3.1-flash-image",
            selection_reason="default",
        )
        data = nb.build_interaction_input(req, "prompt", decision)
        assert data["generation_config"] == {"thinking_level": "high", "seed": 42}


class TestResponseExtraction:
    def test_extract_images(self):
        b64 = base64.b64encode(b"imagebytes").decode()
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "some text"},
                            {"inline_data": {"mime_type": "image/png", "data": b64}},
                        ]
                    }
                }
            ]
        }
        images = nb.extract_images(response)
        assert images == [b"imagebytes"]

    def test_extract_images_empty(self):
        assert nb.extract_images({"candidates": []}) == []
        assert nb.extract_images(None) == []

    def test_extract_grounding_metadata(self):
        response = {
            "candidates": [
                {
                    "grounding_metadata": {
                        "sources": ["https://example.com"],
                        "citations": ["claim"],
                        "suggestions": ["search"],
                    }
                }
            ]
        }
        meta = nb.extract_grounding_metadata(response)
        assert meta == {
            "sources": ["https://example.com"],
            "citations": ["claim"],
            "search_suggestions": ["search"],
        }

    def test_extract_grounding_metadata_none(self):
        assert nb.extract_grounding_metadata({"candidates": []}) is None

    def test_classify_response_error_valid(self):
        b64 = base64.b64encode(b"imagebytes").decode()
        response = {
            "candidates": [{"content": {"parts": [{"inline_data": {"data": b64}}]}}]
        }
        assert nb.classify_response_error(response) is None

    def test_classify_response_error_empty(self):
        assert (
            nb.classify_response_error({"candidates": []}) == nb.ExitCode.EMPTY_RESPONSE
        )

    def test_classify_response_error_safety(self):
        response = {
            "candidates": [{"finish_reason": "SAFETY"}],
        }
        assert nb.classify_response_error(response) == nb.ExitCode.SAFETY_REFUSAL

    def test_is_safety_refusal(self):
        assert nb.is_safety_refusal({"candidates": [{"finish_reason": "SAFETY"}]})
        assert not nb.is_safety_refusal({"candidates": [{"finish_reason": "STOP"}]})


class TestRetryHandler:
    def test_is_retryable(self):
        assert nb.is_retryable(ConnectionError())
        assert nb.is_retryable(TimeoutError())
        assert nb.is_retryable(Exception("429 rate limit"))
        assert nb.is_retryable(Exception("timeout"))
        assert not nb.is_retryable(ValueError("bad request"))

        class ServerError(Exception):
            code = 500

        assert nb.is_retryable(ServerError())

        class ForbiddenError(Exception):
            status_code = 403

        assert not nb.is_retryable(ForbiddenError())

    def test_classify_api_exception(self):
        class QuotaError(Exception):
            code = 429

        class AuthError(Exception):
            status_code = 403

        class BadRequestError(Exception):
            code = 400

        assert nb.classify_api_exception(QuotaError()) == nb.ExitCode.QUOTA_EXCEEDED
        assert nb.classify_api_exception(AuthError()) == nb.ExitCode.AUTH_FAILURE
        assert nb.classify_api_exception(BadRequestError()) == nb.ExitCode.INVALID_ARGS
        assert (
            nb.classify_api_exception(Exception("rate limit"))
            == nb.ExitCode.QUOTA_EXCEEDED
        )
        assert (
            nb.classify_api_exception(Exception("unauthorized"))
            == nb.ExitCode.AUTH_FAILURE
        )
        assert (
            nb.classify_api_exception(Exception("safety block"))
            == nb.ExitCode.SAFETY_REFUSAL
        )
        assert nb.classify_api_exception(Exception("unknown")) == nb.ExitCode.INTERNAL

    def test_with_retry_succeeds_first(self, monkeypatch):
        monkeypatch.setattr(nb.time, "sleep", lambda _s: None)
        monkeypatch.setattr(nb.random, "uniform", lambda _a, _b: 1.0)
        counter = {"n": 0}

        def fn():
            counter["n"] += 1
            return "ok"

        assert nb.with_retry(fn) == "ok"
        assert counter["n"] == 1

    def test_with_retry_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(nb.time, "sleep", lambda _s: None)
        monkeypatch.setattr(nb.random, "uniform", lambda _a, _b: 1.0)
        counter = {"n": 0}

        def fn():
            counter["n"] += 1
            if counter["n"] < 3:
                raise ConnectionError("transient")
            return "ok"

        assert nb.with_retry(fn, max_attempts=3) == "ok"
        assert counter["n"] == 3

    def test_with_retry_gives_up_on_non_retryable(self, monkeypatch):
        monkeypatch.setattr(nb.time, "sleep", lambda _s: None)
        monkeypatch.setattr(nb.random, "uniform", lambda _a, _b: 1.0)
        counter = {"n": 0}

        def fn():
            counter["n"] += 1
            raise ValueError("bad")

        with pytest.raises(ValueError):
            nb.with_retry(fn, max_attempts=3)
        assert counter["n"] == 1

    def test_with_retry_exhausts_attempts(self, monkeypatch):
        monkeypatch.setattr(nb.time, "sleep", lambda _s: None)
        monkeypatch.setattr(nb.random, "uniform", lambda _a, _b: 1.0)
        counter = {"n": 0}

        def fn():
            counter["n"] += 1
            raise ConnectionError("transient")

        with pytest.raises(ConnectionError):
            nb.with_retry(fn, max_attempts=2)
        assert counter["n"] == 2


class TestGeneratePipeline:
    def _make_ctx(self, **kwargs):
        obj = {
            "model": "auto",
            "output_dir": None,
            "api_key": None,
            "config": None,
            "format": "png",
            "json": False,
            "quiet": False,
            "verbose": False,
            "dry_run": False,
            "overwrite": False,
            "seed": None,
            "request_id": None,
        }
        obj.update(kwargs)

        class FakeContext:
            def __init__(self, obj):
                self.obj = obj

        return FakeContext(obj)

    def test_dry_run_does_not_call_api(self, capsys, monkeypatch):
        ctx = self._make_ctx(dry_run=True)
        nb.run_generate_pipeline(ctx, "a test prompt")
        captured = capsys.readouterr()
        assert "gemini-3.1-flash-image" in captured.err
        assert "Dry Run" in captured.err

    def test_dry_run_json(self, capsys, monkeypatch):
        ctx = self._make_ctx(dry_run=True, json=True)
        nb.run_generate_pipeline(ctx, "a test prompt")
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["status"] == "dry-run"
        assert payload["outputs"][0]["count"] == 1

    def test_missing_api_key_fails(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        ctx = self._make_ctx()
        with pytest.raises(nb.typer.Exit) as exc:
            nb.run_generate_pipeline(ctx, "test")
        assert exc.value.exit_code == nb.ExitCode.AUTH_FAILURE

    def test_invalid_size_model_combo_fails(self):
        ctx = self._make_ctx(model="lite")
        with pytest.raises(nb.typer.Exit) as exc:
            nb.run_generate_pipeline(ctx, "test", size="4K")
        assert exc.value.exit_code == nb.ExitCode.CAPABILITY_MISMATCH

    def test_show_estimate(self, capsys):
        ctx = self._make_ctx()
        nb.run_generate_pipeline(ctx, "test", show_estimate=True)
        captured = capsys.readouterr()
        assert "Estimated cost" in captured.out

    def test_max_estimated_cost_exceeded(self):
        ctx = self._make_ctx()
        with pytest.raises(nb.typer.Exit) as exc:
            nb.run_generate_pipeline(ctx, "test", max_estimated_cost=0.01)
        assert exc.value.exit_code == nb.ExitCode.INVALID_ARGS

    def test_preset_applies_and_cli_overrides(self, capsys):
        ctx = self._make_ctx(dry_run=True)
        nb.run_generate_pipeline(ctx, "test", size="4K", preset_name="photo")
        captured = capsys.readouterr()
        # Photo preset sets aspect=3:2, size=2K, model=flash; CLI overrides size to 4K.
        assert "3:2" in captured.err
        assert "4K" in captured.err
        assert "gemini-3.1-flash-image" in captured.err

    def test_successful_run_writes_image_and_manifest(self, tmp_path, monkeypatch):
        image_bytes = b"\x89PNG\x00\x00"
        b64 = base64.b64encode(image_bytes).decode()
        fake_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inline_data": {"mime_type": "image/png", "data": b64}}
                        ]
                    }
                }
            ]
        }
        calls = []

        class FakeClient:
            class interactions:
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    return fake_response

        monkeypatch.setattr(nb, "genai", nb.genai)
        monkeypatch.setattr(nb.genai, "Client", lambda api_key: FakeClient())
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(nb.time, "sleep", lambda _s: None)

        output_dir = tmp_path / "out"
        ctx = self._make_ctx(output_dir=output_dir)
        nb.run_generate_pipeline(ctx, "test output")

        assert len(calls) == 1
        assert calls[0]["model"] == "gemini-3.1-flash-image"
        outputs = list(output_dir.iterdir())
        assert len(outputs) == 2  # image + manifest
        manifest_path = next(p for p in outputs if p.suffix == ".json")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema_version"] == "1.0"
        assert manifest["outputs"][0]["sha256"] == nb.sha256_bytes(image_bytes)

    def test_empty_response_fails(self, monkeypatch):
        class FakeClient:
            class interactions:
                @staticmethod
                def create(**kwargs):
                    return {"candidates": []}

        monkeypatch.setattr(nb.genai, "Client", lambda api_key: FakeClient())
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(nb.time, "sleep", lambda _s: None)
        ctx = self._make_ctx()
        with pytest.raises(nb.typer.Exit) as exc:
            nb.run_generate_pipeline(ctx, "test")
        assert exc.value.exit_code == nb.ExitCode.EMPTY_RESPONSE

    def test_safety_refusal_fails(self, monkeypatch):
        class FakeClient:
            class interactions:
                @staticmethod
                def create(**kwargs):
                    return {"candidates": [{"finish_reason": "SAFETY"}]}

        monkeypatch.setattr(nb.genai, "Client", lambda api_key: FakeClient())
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(nb.time, "sleep", lambda _s: None)
        ctx = self._make_ctx()
        with pytest.raises(nb.typer.Exit) as exc:
            nb.run_generate_pipeline(ctx, "test")
        assert exc.value.exit_code == nb.ExitCode.SAFETY_REFUSAL

    def test_retry_with_pro_escalates(self, tmp_path, monkeypatch):
        image_bytes = b"\x89PNG\x00\x00"
        b64 = base64.b64encode(image_bytes).decode()
        pro_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inline_data": {"mime_type": "image/png", "data": b64}}
                        ]
                    }
                }
            ]
        }

        class FakeClient:
            class interactions:
                calls = []

                @staticmethod
                def create(**kwargs):
                    FakeClient.interactions.calls.append(kwargs)
                    if kwargs["model"] == "gemini-3.1-flash-image":
                        return {"candidates": []}
                    return pro_response

        monkeypatch.setattr(nb.genai, "Client", lambda api_key: FakeClient())
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(nb.time, "sleep", lambda _s: None)

        output_dir = tmp_path / "out"
        ctx = self._make_ctx(output_dir=output_dir)
        nb.run_generate_pipeline(ctx, "test", retry_with_pro=True)

        models = [c["model"] for c in FakeClient.interactions.calls]
        assert models == ["gemini-3.1-flash-image", "gemini-3-pro-image"]
        manifest_path = next(p for p in output_dir.iterdir() if p.suffix == ".json")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["model"]["resolved"] == "gemini-3-pro-image"

    def test_prompt_file(self, tmp_path, capsys):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("file prompt")
        ctx = self._make_ctx(dry_run=True)
        nb.run_generate_pipeline(ctx, None, prompt_file=prompt_file)
        captured = capsys.readouterr()
        assert "file prompt" in captured.err

    def test_missing_prompt_and_prompt_file_fails(self):
        ctx = self._make_ctx()
        with pytest.raises(nb.typer.Exit) as exc:
            nb.run_generate_pipeline(ctx, None)
        assert exc.value.exit_code == nb.ExitCode.INVALID_ARGS
