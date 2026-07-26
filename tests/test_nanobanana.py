import base64
import json
import os
import re
import time
from pathlib import Path
from unittest.mock import MagicMock

import conftest  # type: ignore[import]
import pytest
import typer
from conftest import (  # type: ignore[import]
    COST_TABLE,
    ExitCode,
    GeneratedAsset,
    ImageRequest,
    ModelDecision,
    ReferenceImage,
    _should_use_lite,
    _should_use_pro,
    annotate_references,
    atomic_write,
    build_edit_instruction,
    build_interaction_input,
    build_normalized_prompt,
    classify_api_exception,
    detect_mime_type,
    estimate_cost,
    execute_request,
    expand_negative,
    extract_grounding_metadata,
    generate_filename,
    get_capability,
    is_retryable,
    is_safety_refusal,
    resolve_model_alias,
    resolve_output_path,
    run_generate_pipeline,
    select_model,
    sha256_bytes,
    sha256_file,
    slugify,
    supports,
    validate_request,
    with_retry,
    write_manifest,
)

# =============================================================================
# Helpers (v2 SDK format)
# =============================================================================


def make_image_response(image_data: bytes) -> dict:
    b64 = base64.b64encode(image_data).decode("ascii")
    return {
        "output_image": {"data": b64, "mime_type": "image/png"},
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {"type": "image", "data": b64, "mime_type": "image/png"}
                ],
            }
        ],
    }


def make_context(**overrides):
    ctx = MagicMock()
    ctx.obj = {
        "model": "auto",
        "output_dir": None,
        "api_key": "test-key",
        "config": None,
        "format": "png",
        "json": False,
        "quiet": True,
        "verbose": False,
        "dry_run": False,
        "overwrite": False,
        "seed": None,
        "request_id": None,
    }
    ctx.obj.update(overrides)
    return ctx


# =============================================================================
# Unit Tests
# =============================================================================


class TestModelSelection:
    def test_auto_simple_defaults_to_flash(self):
        request = ImageRequest(command="generate", prompt="a banana", model="auto")
        decision = select_model(request)
        assert decision.resolved == "gemini-3.1-flash-image"
        assert "default general-purpose" in decision.selection_reason

    def test_auto_draft_1k_to_lite(self):
        request = ImageRequest(
            command="generate",
            prompt="a banana",
            model="auto",
            image_size="1K",
            quality="draft",
        )
        decision = select_model(request)
        assert decision.resolved == "gemini-3.1-flash-lite-image"

    def test_auto_variations_to_lite(self):
        request = ImageRequest(
            command="variations", prompt="a banana", model="auto", image_size="1K"
        )
        decision = select_model(request)
        assert decision.resolved == "gemini-3.1-flash-lite-image"

    def test_auto_diagram_to_pro(self):
        request = ImageRequest(command="diagram", prompt="a system diagram", model="auto")
        decision = select_model(request)
        assert decision.resolved == "gemini-3-pro-image"

    def test_auto_product_to_pro(self):
        request = ImageRequest(command="product", prompt="a product shot", model="auto")
        decision = select_model(request)
        assert decision.resolved == "gemini-3-pro-image"

    def test_auto_explicit_text_long_prompt_to_pro(self):
        prompt = (
            'Create an image with the exact spelled-out headline "Welcome to the Future" '
            "in large letters on a billboard"
        )
        request = ImageRequest(command="generate", prompt=prompt, model="auto")
        decision = select_model(request)
        assert decision.resolved == "gemini-3-pro-image"

    def test_explicit_flash(self):
        request = ImageRequest(command="generate", prompt="a banana", model="flash")
        decision = select_model(request)
        assert decision.resolved == "gemini-3.1-flash-image"
        assert decision.requested == "flash"

    def test_explicit_lite(self):
        request = ImageRequest(command="generate", prompt="a banana", model="lite")
        decision = select_model(request)
        assert decision.resolved == "gemini-3.1-flash-lite-image"

    def test_explicit_pro(self):
        request = ImageRequest(command="generate", prompt="a banana", model="pro")
        decision = select_model(request)
        assert decision.resolved == "gemini-3-pro-image"

    def test_should_use_lite_false_for_4k(self):
        request = ImageRequest(
            command="generate",
            prompt="x",
            model="auto",
            image_size="4K",
            quality="draft",
        )
        assert _should_use_lite(request) is False

    def test_should_use_lite_false_for_grounding(self):
        request = ImageRequest(
            command="generate",
            prompt="x",
            model="auto",
            image_size="1K",
            quality="draft",
            grounding="web",
        )
        assert _should_use_lite(request) is False

    def test_should_use_pro_professional_quality(self):
        request = ImageRequest(
            command="generate",
            prompt="x",
            model="auto",
            quality="professional",
        )
        assert _should_use_pro(request) is True

    def test_should_use_pro_infographic_command(self):
        request = ImageRequest(command="infographic", prompt="x", model="auto")
        assert _should_use_pro(request) is True

    def test_resolve_model_alias_maps_full_name(self):
        assert resolve_model_alias("gemini-3.1-flash-image") == "gemini-3.1-flash-image"

    def test_resolve_model_alias_unknown_fails(self):
        with pytest.raises(typer.Exit) as exc:
            resolve_model_alias("unknown")
        assert exc.value.exit_code == ExitCode.INVALID_ARGS

    def test_get_capability_and_supports(self):
        assert get_capability("flash", "grounding") is True
        assert supports("flash", "grounding") is True
        assert supports("lite", "grounding") is False


class TestValidation:
    def test_lite_rejects_4k(self):
        request = ImageRequest(command="generate", prompt="x", model="lite", image_size="4K")
        with pytest.raises(typer.Exit) as exc:
            validate_request(request)
        assert exc.value.exit_code == ExitCode.CAPABILITY_MISMATCH

    def test_flash_allows_2k(self):
        request = ImageRequest(command="generate", prompt="x", model="flash", image_size="2K")
        assert validate_request(request) == []

    def test_lite_rejects_grounding(self):
        request = ImageRequest(
            command="generate",
            prompt="x",
            model="lite",
            image_size="1K",
            grounding="web",
        )
        with pytest.raises(typer.Exit) as exc:
            validate_request(request)
        assert exc.value.exit_code == ExitCode.CAPABILITY_MISMATCH

    def test_invalid_aspect_ratio(self):
        request = ImageRequest(command="generate", prompt="x", model="flash", aspect_ratio="99:1")
        with pytest.raises(typer.Exit) as exc:
            validate_request(request)
        assert exc.value.exit_code == ExitCode.CAPABILITY_MISMATCH

    def test_flash_only_ratio_on_lite(self):
        request = ImageRequest(
            command="generate",
            prompt="x",
            model="lite",
            aspect_ratio="1:4",
            image_size="1K",
        )
        with pytest.raises(typer.Exit) as exc:
            validate_request(request)
        assert exc.value.exit_code == ExitCode.CAPABILITY_MISMATCH

    def test_too_many_references(self):
        refs = tuple(
            ReferenceImage(
                path=Path("dummy.png"),
                role="reference",
                mime_type="image/png",
                sha256="a",
            )
            for _ in range(12)
        )
        request = ImageRequest(command="generate", prompt="x", model="flash", references=refs)
        with pytest.raises(typer.Exit) as exc:
            validate_request(request)
        assert exc.value.exit_code == ExitCode.CAPABILITY_MISMATCH

    def test_allow_degraded_returns_warnings(self):
        refs = tuple(
            ReferenceImage(
                path=Path("dummy.png"),
                role="reference",
                mime_type="image/png",
                sha256="a",
            )
            for _ in range(12)
        )
        request = ImageRequest(command="generate", prompt="x", model="flash", references=refs)
        warnings = validate_request(request, allow_degraded=True)
        assert len(warnings) == 1
        assert "exceed" in warnings[0]

    def test_pro_thinking_unavailable(self):
        request = ImageRequest(
            command="generate",
            prompt="x",
            model="pro",
            image_size="1K",
            thinking_level="high",
        )
        with pytest.raises(typer.Exit) as exc:
            validate_request(request)
        assert exc.value.exit_code == ExitCode.CAPABILITY_MISMATCH

    def test_flash_only_ratio_on_pro(self):
        request = ImageRequest(
            command="generate",
            prompt="x",
            model="pro",
            aspect_ratio="1:4",
            image_size="1K",
        )
        with pytest.raises(typer.Exit) as exc:
            validate_request(request)
        assert exc.value.exit_code == ExitCode.CAPABILITY_MISMATCH


class TestPromptAssembly:
    def test_required_sections_present(self):
        request = ImageRequest(command="generate", prompt="a banana", model="auto")
        prompt = build_normalized_prompt(request)
        assert "TASK:" in prompt
        assert "REQUIREMENTS:" in prompt
        assert "AVOID:" in prompt
        assert "OUTPUT INTENT:" in prompt

    def test_negative_converted_to_avoid(self):
        request = ImageRequest(command="generate", prompt="a banana", model="auto")
        prompt = build_normalized_prompt(request, negative="blur, noise")
        assert "- blur" in prompt
        assert "- noise" in prompt

    def test_preset_prefix_present(self):
        request = ImageRequest(command="generate", prompt="a banana", model="auto")
        preset = conftest.PRESETS["icon"]
        prompt = build_normalized_prompt(request, preset=preset)
        assert preset["prompt_prefix"] in prompt
        assert "CONTEXT:" in prompt

    def test_default_avoids_present(self):
        request = ImageRequest(command="generate", prompt="a banana", model="auto")
        prompt = build_normalized_prompt(request)
        assert "Illegible labels" in prompt
        assert "Generic or off-brand" in prompt

    def test_references_in_prompt(self):
        ref = ReferenceImage(path=Path("x.png"), role="style", mime_type="image/png", sha256="abc")
        request = ImageRequest(command="compose", prompt="blend", model="auto", references=(ref,))
        prompt = build_normalized_prompt(request)
        assert "REFERENCE ROLES:" in prompt
        assert "Reference 1 (style)" in prompt

    def test_preserve_section_for_edit(self):
        request = ImageRequest(command="edit", prompt="change color", model="auto")
        prompt = build_normalized_prompt(request)
        assert "PRESERVE:" in prompt

    def test_annotate_references_empty(self):
        assert annotate_references(()) == ""

    def test_expand_negative_empty(self):
        assert expand_negative("") == ""
        assert expand_negative("   ") == ""

    def test_output_intent_with_text(self):
        request = ImageRequest(command="generate", prompt="x", model="auto", text_output=True)
        prompt = build_normalized_prompt(request)
        assert "text explanation" in prompt


class TestFilenames:
    def test_filename_format(self):
        filename = generate_filename("banana", index=1, extension="png")
        assert re.match(r"^\d{8}-\d{6}-banana-01\.png$", filename)

    def test_sanitization(self):
        filename = generate_filename("Hello World!!!", index=2, extension="jpeg")
        assert "hello-world" in filename

    def test_truncation(self):
        long_slug = "a" * 100
        filename = generate_filename(long_slug, index=1, extension="png")
        slug_part = filename.split("-")[2]
        assert len(slug_part) <= 64

    def test_slugify(self):
        assert slugify("The Quick Brown Fox") == "the-quick-brown"

    def test_slugify_empty(self):
        assert slugify("!!!") == "image"

    def test_resolve_output_path(self, temp_dir):
        request = ImageRequest(command="generate", prompt="a banana", model="auto")
        path = resolve_output_path(request, None, temp_dir, "png")
        assert path.parent == temp_dir
        assert path.suffix == ".png"

    def test_resolve_output_path_specified(self, temp_dir):
        request = ImageRequest(command="generate", prompt="a banana", model="auto")
        specified = temp_dir / "custom.png"
        assert resolve_output_path(request, specified, None, "png") == specified


class TestMime:
    def test_png(self, temp_dir):
        path = temp_dir / "image.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        assert detect_mime_type(path) == "image/png"

    def test_jpeg(self, temp_dir):
        path = temp_dir / "image.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
        assert detect_mime_type(path) == "image/jpeg"

    def test_unknown(self, temp_dir):
        path = temp_dir / "image.gif"
        path.write_bytes(b"GIF89a" + b"\x00" * 20)
        with pytest.raises(typer.Exit) as exc:
            detect_mime_type(path)
        assert exc.value.exit_code == ExitCode.INPUT_FAILURE

    def test_missing_file(self, temp_dir):
        path = temp_dir / "missing.png"
        with pytest.raises(typer.Exit) as exc:
            detect_mime_type(path)
        assert exc.value.exit_code == ExitCode.INPUT_FAILURE


class TestSha256:
    def test_bytes_known_value(self):
        data = b"abc"
        expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        assert sha256_bytes(data) == expected

    def test_file_known_value(self, temp_dir):
        path = temp_dir / "abc.txt"
        path.write_bytes(b"abc")
        expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        assert sha256_file(path) == expected


class TestManifest:
    def test_required_fields(self, temp_dir):
        request = ImageRequest(command="generate", prompt="a banana", model="flash")
        decision = ModelDecision(
            requested="flash",
            resolved="gemini-3.1-flash-image",
            selection_reason="explicit",
        )
        asset = GeneratedAsset(path=temp_dir / "out.png", mime_type="image/png", sha256="abc")
        manifest_path = write_manifest(request, decision, "normalized", [asset], temp_dir)
        data = json.loads(manifest_path.read_text())
        assert data["schema_version"] == "1.0"
        assert data["run_id"] == request.request_id
        assert data["model"]["resolved"] == decision.resolved
        assert data["generation"]["normalized_prompt"] == "normalized"
        assert data["outputs"][0]["sha256"] == "abc"

    def test_schema_version(self, temp_dir):
        request = ImageRequest(command="generate", prompt="x", model="flash")
        decision = ModelDecision(
            requested="flash",
            resolved="gemini-3.1-flash-image",
            selection_reason="explicit",
        )
        asset = GeneratedAsset(path=temp_dir / "out.png", mime_type="image/png", sha256="abc")
        manifest_path = write_manifest(request, decision, "norm", [asset], temp_dir)
        data = json.loads(manifest_path.read_text())
        assert data["schema_version"] == "1.0"

    def test_warnings_included(self, temp_dir):
        request = ImageRequest(command="generate", prompt="x", model="flash")
        decision = ModelDecision(
            requested="flash",
            resolved="gemini-3.1-flash-image",
            selection_reason="explicit",
        )
        asset = GeneratedAsset(path=temp_dir / "out.png", mime_type="image/png", sha256="abc")
        manifest_path = write_manifest(
            request, decision, "norm", [asset], temp_dir, warnings=["warn1"]
        )
        data = json.loads(manifest_path.read_text())
        assert data["warnings"] == ["warn1"]

    def test_grounding_metadata_included(self, temp_dir):
        request = ImageRequest(command="generate", prompt="x", model="flash")
        decision = ModelDecision(
            requested="flash",
            resolved="gemini-3.1-flash-image",
            selection_reason="explicit",
        )
        asset = GeneratedAsset(path=temp_dir / "out.png", mime_type="image/png", sha256="abc")
        grounding = {"sources": ["https://example.com"]}
        manifest_path = write_manifest(
            request,
            decision,
            "norm",
            [asset],
            temp_dir,
            grounding_metadata=grounding,
        )
        data = json.loads(manifest_path.read_text())
        assert data["sources"] == grounding


class TestAtomicWrite:
    def test_create(self, temp_dir):
        path = temp_dir / "out.txt"
        atomic_write(path, b"hello")
        assert path.read_bytes() == b"hello"

    def test_existing_error(self, temp_dir):
        path = temp_dir / "out.txt"
        path.write_bytes(b"existing")
        with pytest.raises(FileExistsError):
            atomic_write(path, b"new")

    def test_overwrite(self, temp_dir):
        path = temp_dir / "out.txt"
        path.write_bytes(b"existing")
        atomic_write(path, b"new", overwrite=True)
        assert path.read_bytes() == b"new"

    def test_parent_directory_created(self, temp_dir):
        path = temp_dir / "nested" / "out.txt"
        atomic_write(path, b"data")
        assert path.read_bytes() == b"data"


class TestRetry:
    def test_429_retryable(self):
        class Exc(Exception):
            code = "429"

        assert is_retryable(Exc("rate limited")) is True

    def test_500_retryable(self):
        class Exc(Exception):
            code = 500

        assert is_retryable(Exc("server error")) is True

    def test_400_not_retryable(self):
        class Exc(Exception):
            code = 400

        assert is_retryable(Exc("bad request")) is False

    def test_401_not_retryable(self):
        class Exc(Exception):
            code = 401

        assert is_retryable(Exc("unauthorized")) is False

    def test_safety_refusal_check(self):
        response = {"status": "SAFETY"}
        assert is_safety_refusal(response) is True

    def test_safety_refusal_prompt_feedback(self):
        response = {"steps": [{"finish_reason": "SAFETY"}]}
        assert is_safety_refusal(response) is True

    def test_with_retry_success_after_transient(self, monkeypatch):
        class Transient(Exception):
            code = "429"

        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 2:
                raise Transient("retry")
            return "ok"

        monkeypatch.setattr(time, "sleep", lambda _s: None)
        assert with_retry(fn, max_attempts=3) == "ok"
        assert len(calls) == 2

    def test_with_retry_exhausted(self, monkeypatch):
        class Transient(Exception):
            code = "500"

        def fn():
            raise Transient("fail")

        monkeypatch.setattr(time, "sleep", lambda _s: None)
        with pytest.raises(Transient):
            with_retry(fn, max_attempts=2)

    def test_classify_api_exception_quota(self):
        class QuotaError(Exception):
            code = 429

        assert classify_api_exception(QuotaError()) == ExitCode.QUOTA_EXCEEDED

    def test_classify_api_exception_auth(self):
        class AuthError(Exception):
            status_code = 403

        assert classify_api_exception(AuthError()) == ExitCode.AUTH_FAILURE

    def test_execute_request_retries_then_succeeds(self, monkeypatch):
        class RateLimit(Exception):
            code = "429"

        calls = []

        def create(**kwargs):
            calls.append(1)
            if len(calls) < 2:
                raise RateLimit("rate limit")
            return "ok"

        client = MagicMock()
        client.interactions.create.side_effect = create
        monkeypatch.setattr(time, "sleep", lambda _s: None)
        assert execute_request(client, {}) == "ok"
        assert len(calls) == 2


class TestCost:
    def test_flash_2k(self):
        assert estimate_cost("flash", "2K") == COST_TABLE["gemini-3.1-flash-image"]["2K"]

    def test_lite_1k(self):
        assert estimate_cost("lite", "1K") == COST_TABLE["gemini-3.1-flash-lite-image"]["1K"]

    def test_unknown_size(self):
        assert estimate_cost("flash", "8K") is None

    def test_count_multiplier(self):
        unit = COST_TABLE["gemini-3.1-flash-image"]["1K"]
        assert estimate_cost("flash", "1K", count=3) == unit * 3

    def test_auto_returns_none(self):
        assert estimate_cost("auto", "1K") is None


class TestSecretRedaction:
    def test_key_not_in_manifest(self, temp_dir, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "secret-key-123")
        request = ImageRequest(command="generate", prompt="x", model="flash")
        decision = ModelDecision(
            requested="flash",
            resolved="gemini-3.1-flash-image",
            selection_reason="explicit",
        )
        asset = GeneratedAsset(path=temp_dir / "out.png", mime_type="image/png", sha256="abc")
        manifest_path = write_manifest(request, decision, "norm", [asset], temp_dir)
        text = manifest_path.read_text()
        assert "secret-key-123" not in text

    def test_key_not_in_log(self, temp_dir, monkeypatch, capsys, mock_client):
        monkeypatch.setenv("GEMINI_API_KEY", "secret-key-123")
        image_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        mock_client.Client.return_value.interactions.create.return_value = make_image_response(
            image_data
        )
        monkeypatch.setattr(time, "sleep", lambda _s: None)

        ctx = make_context(output_dir=temp_dir, api_key=None)
        run_generate_pipeline(
            ctx,
            "a banana",
            aspect="1:1",
            size="1K",
            output=temp_dir / "out.png",
        )

        captured = capsys.readouterr()
        assert "secret-key-123" not in captured.err
        assert "secret-key-123" not in captured.out


class TestBuildEditInstruction:
    def test_all_sections(self):
        result = build_edit_instruction(
            "change background",
            preserve="person",
            change="background to blue",
            mask="semantic",
            strict_preservation=True,
        )
        assert "change background" in result
        assert "PRESERVE:" in result
        assert "person" in result
        assert "CHANGE:" in result
        assert "background to blue" in result
        assert "MASK:" in result
        assert "STRICT PRESERVATION:" in result

    def test_no_mask(self):
        result = build_edit_instruction("change bg")
        assert "MASK:" not in result

    def test_mask_none(self):
        result = build_edit_instruction("change bg", mask="none")
        assert "MASK:" not in result


class TestExtractGroundingMetadata:
    def test_extracts_sources(self):
        response = {
            "steps": [
                {
                    "type": "model_output",
                    "grounding_metadata": {
                        "sources": ["https://example.com"],
                        "citations": ["claim"],
                        "suggestions": ["search"],
                    },
                }
            ]
        }
        meta = extract_grounding_metadata(response)
        assert meta == {
            "sources": ["https://example.com"],
            "citations": ["claim"],
            "search_suggestions": ["search"],
        }

    def test_empty_returns_none(self):
        assert extract_grounding_metadata({"steps": []}) is None


# =============================================================================
# Contract Tests
# =============================================================================


class TestContract:
    def test_successful_response(self, temp_dir, monkeypatch, mock_client):
        image_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        mock_client.Client.return_value.interactions.create.return_value = make_image_response(
            image_data
        )
        monkeypatch.setattr(time, "sleep", lambda _s: None)

        run_generate_pipeline(
            make_context(output_dir=temp_dir),
            "a banana",
            aspect="1:1",
            size="1K",
            output=temp_dir / "out.png",
        )
        assert (temp_dir / "out.png").exists()
        assert (temp_dir / "out.png.manifest.json").exists()

    def test_safety_refusal(self, temp_dir, monkeypatch, mock_client):
        mock_client.Client.return_value.interactions.create.return_value = {
            "status": "SAFETY",
            "steps": [],
        }
        monkeypatch.setattr(time, "sleep", lambda _s: None)

        with pytest.raises(typer.Exit) as exc:
            run_generate_pipeline(
                make_context(output_dir=temp_dir),
                "a banana",
                aspect="1:1",
                size="1K",
            )
        assert exc.value.exit_code == ExitCode.SAFETY_REFUSAL

    def test_auth_failure(self, temp_dir, monkeypatch, mock_client):
        class AuthError(Exception):
            code = "401"

        mock_client.Client.return_value.interactions.create.side_effect = AuthError("unauthorized")
        monkeypatch.setattr(time, "sleep", lambda _s: None)

        with pytest.raises(typer.Exit) as exc:
            run_generate_pipeline(
                make_context(output_dir=temp_dir),
                "a banana",
                aspect="1:1",
                size="1K",
            )
        assert exc.value.exit_code == ExitCode.AUTH_FAILURE

    def test_empty_response(self, temp_dir, monkeypatch, mock_client):
        mock_client.Client.return_value.interactions.create.return_value = {
            "steps": [],
        }
        monkeypatch.setattr(time, "sleep", lambda _s: None)

        with pytest.raises(typer.Exit) as exc:
            run_generate_pipeline(
                make_context(output_dir=temp_dir),
                "a banana",
                aspect="1:1",
                size="1K",
            )
        assert exc.value.exit_code == ExitCode.EMPTY_RESPONSE

    def test_rate_limit_retry(self, temp_dir, monkeypatch, mock_client):
        image_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        response = make_image_response(image_data)

        class RateLimit(Exception):
            code = "429"

        calls = []

        def create(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RateLimit("rate limit")
            return response

        mock_client.Client.return_value.interactions.create.side_effect = create
        monkeypatch.setattr(time, "sleep", lambda _s: None)

        run_generate_pipeline(
            make_context(output_dir=temp_dir),
            "a banana",
            aspect="1:1",
            size="1K",
            output=temp_dir / "out.png",
        )
        assert len(calls) == 2
        assert (temp_dir / "out.png").exists()

    def test_json_output_response_format(self):
        request = ImageRequest(
            command="generate",
            prompt="a banana",
            model="flash",
            text_output=True,
            mime_type="image/png",
        )
        input_data = build_interaction_input(
            request,
            "normalized",
            ModelDecision("auto", "gemini-3.1-flash-image", "x"),
        )
        assert input_data["response_format"] == [
            {"type": "text"},
            {"type": "image", "mime_type": "image/png"},
        ]


# =============================================================================
# Live Tests
# =============================================================================


class TestLive:
    @pytest.mark.skipif(
        os.environ.get("NANOBANANA_LIVE_TESTS") != "1",
        reason="Live tests require NANOBANANA_LIVE_TESTS=1",
    )
    def test_live_lite_icon(self, temp_dir):
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            conftest.app,
            [
                "--model",
                "lite",
                "--output-dir",
                str(temp_dir),
                "generate",
                "a simple cog icon",
                "--preset",
                "icon",
            ],
        )
        assert result.exit_code == 0
        outputs = list(temp_dir.glob("*.png"))
        assert len(outputs) >= 1

    @pytest.mark.skipif(
        os.environ.get("NANOBANANA_LIVE_TESTS") != "1",
        reason="Live tests require NANOBANANA_LIVE_TESTS=1",
    )
    def test_live_invalid_combo_rejected(self, temp_dir):
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            conftest.app,
            [
                "--model",
                "lite",
                "--output-dir",
                str(temp_dir),
                "generate",
                "x",
                "--size",
                "4K",
            ],
        )
        assert result.exit_code == ExitCode.CAPABILITY_MISMATCH

    @pytest.mark.skipif(
        os.environ.get("NANOBANANA_LIVE_TESTS") != "1",
        reason="Live tests require NANOBANANA_LIVE_TESTS=1",
    )
    def test_live_flash_2k(self, temp_dir):
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            conftest.app,
            [
                "--model",
                "flash",
                "--output-dir",
                str(temp_dir),
                "generate",
                "a banana",
                "--size",
                "2K",
            ],
        )
        assert result.exit_code == 0
        outputs = list(temp_dir.glob("*.png"))
        assert len(outputs) >= 1
