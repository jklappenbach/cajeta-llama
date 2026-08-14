#!/usr/bin/env bash
# Build + run the cajeta-llama unit tests.
#
# The suite lives under src/test/cajeta and is driven by cajeta-unit's reflective
# @Test discovery (dev.cajeta.unit.Runner). It compiles ONLY the test sources into
# an executable, with the llama library and cajeta-unit supplied as .cja
# classpath dependencies — the compiler links their bitcode into the test binary.
#
# Override paths via env:
#   CAJETA    — compiler binary (default: cajeta on PATH). The loader needs
#               MappedFile + the int64 file path (cajeta main ≥ 2026-08-13);
#               until that ships in a release, point CAJETA at a main build.
#   UNIT_REPO — path to the cajeta-unit checkout (default: ../cajeta-unit)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
CAJETA="${CAJETA:-cajeta}"
UNIT_REPO="${UNIT_REPO:-$here/../cajeta-unit}"

out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

# cajeta-unit resolution (the cajeta-ml pattern), in order:
#   1. $UNIT_CJA        — explicit archive path, used verbatim
#   2. $UNIT_REPO       — sibling checkout when it exists: build it and use
#                         whatever version it emits (local dev, unit HEAD)
#   3. $OLLA_HOME store — an installed dev.cajeta.unit at the version pinned
#                         in cajeta.json's dev-dependencies
#   4. Olla registry    — /v2/resolve + /v2/blob, sha256-verified, cached
#                         under build/. The CI flow: bare runners have no
#                         checkout.
OLLA_HOME="${OLLA_HOME:-$HOME/.olla}"
OLLA_URL="${OLLA_URL:-https://olla.cajeta.dev}"
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1;
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
unit_cja="${UNIT_CJA:-}"
if [[ -z "$unit_cja" && -d "$UNIT_REPO" ]]; then
    echo ">> building cajeta-unit from checkout ($UNIT_REPO)"
    ( cd "$UNIT_REPO" && "$CAJETA" build >/dev/null )
    unit_cja="$(ls -t "$UNIT_REPO"/build/archive/dev.cajeta.unit-*.cja 2>/dev/null | head -1)"
fi
if [[ -z "$unit_cja" ]]; then
    UNIT_VER="$(sed -n 's/.*"dev\.cajeta\.unit"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        "$here/cajeta.json" | head -1)"
    [[ -n "$UNIT_VER" ]] || { echo "no dev.cajeta.unit pin in cajeta.json" >&2; exit 1; }
    store_cja="$OLLA_HOME/dev.cajeta.unit/$UNIT_VER/dev.cajeta.unit-$UNIT_VER.cja"
    cache_cja="$here/build/.unit-cache/dev.cajeta.unit-$UNIT_VER.cja"
    if [[ -f "$store_cja" ]]; then unit_cja="$store_cja"
    elif [[ -f "$cache_cja" ]]; then unit_cja="$cache_cja"
    else
        echo ">> fetching dev.cajeta.unit $UNIT_VER from $OLLA_URL"
        meta="$(curl -fsS "$OLLA_URL/v2/resolve?name=dev.cajeta.unit&version=$UNIT_VER")"
        sha="$(printf '%s' "$meta" | sed -n 's/.*"sha256":"sha256:\([0-9a-f]*\)".*/\1/p')"
        [[ -n "$sha" ]] || { echo "/v2/resolve gave no sha256" >&2; exit 1; }
        mkdir -p "$(dirname "$cache_cja")"
        curl -fsS -o "$cache_cja" "$OLLA_URL/v2/blob/$sha"
        got="$(sha256_of "$cache_cja")"
        [[ "$got" == "$sha" ]] || { rm -f "$cache_cja"; echo "sha256 mismatch fetching unit" >&2; exit 1; }
        unit_cja="$cache_cja"
    fi
fi
[[ -f "$unit_cja" ]] || { echo "could not resolve a dev.cajeta.unit archive" >&2; exit 1; }
echo ">> cajeta-unit: $unit_cja"

# dev.cajeta.codec (ProtobufCursor for raw tokenizer.model, spec 7.10):
# sibling checkout first (the cajeta-unit pattern), then store, then a
# sha256-verified Olla fetch.
CODEC_VER="$(sed -n 's/.*"dev\.cajeta\.codec"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$here/cajeta.json" | head -1)"
CODEC_REPO="${CODEC_REPO:-$here/../cajeta-codec}"
codec_cja=""
if [[ -d "$CODEC_REPO" ]]; then
    echo ">> building dev.cajeta.codec from checkout ($CODEC_REPO)"
    ( cd "$CODEC_REPO" && "$CAJETA" build >/dev/null )
    codec_cja="$(ls -t "$CODEC_REPO"/build/archive/dev.cajeta.codec-*.cja 2>/dev/null | head -1)"
fi
if [[ -z "$codec_cja" ]]; then
    codec_cja="$OLLA_HOME/dev.cajeta.codec/$CODEC_VER/dev.cajeta.codec-$CODEC_VER.cja"
fi
if [[ ! -f "$codec_cja" ]]; then
    codec_cja="$here/build/.unit-cache/dev.cajeta.codec-$CODEC_VER.cja"
    if [[ ! -f "$codec_cja" ]]; then
        echo ">> fetching dev.cajeta.codec $CODEC_VER from $OLLA_URL"
        meta="$(curl -fsS "$OLLA_URL/v2/resolve?name=dev.cajeta.codec&version=$CODEC_VER")"
        sha="$(printf '%s' "$meta" | sed -n 's/.*"sha256":"sha256:\([0-9a-f]*\)".*/\1/p')"
        [[ -n "$sha" ]] || { echo "/v2/resolve gave no sha256 for codec" >&2; exit 1; }
        mkdir -p "$(dirname "$codec_cja")"
        curl -fsS -o "$codec_cja" "$OLLA_URL/v2/blob/$sha"
        got="$(sha256_of "$codec_cja")"
        [[ "$got" == "$sha" ]] || { rm -f "$codec_cja"; echo "sha256 mismatch fetching codec" >&2; exit 1; }
    fi
fi
echo ">> dev.cajeta.codec: $codec_cja"

echo ">> building llama library .cja"
"$CAJETA" --emit=cja -o "$out/llama.cja" \
    --classpath="$codec_cja" \
    dev.cajeta.llama.Llama.run "$here/src/main/cajeta" "$out" >/dev/null

echo ">> building + running the test binary"
# --xpu-backend=cpu: the engine's device paths (device-resident weight loads,
# later the decode kernels) are exercised on the portable CPU backend, the
# PlacementDispatchTests discipline — real KernelBuffers, no silicon needed.
"$CAJETA" --emit=exe --profile=test --xpu-backend=cpu \
    --classpath="$out/llama.cja,$unit_cja,$codec_cja" \
    -o "$out/llamatests" \
    dev.cajeta.llama.selftest.TestMain.run "$here/src/test/cajeta" "$out" >/dev/null

"$out/llamatests"

echo ">> building + running the test binary under --release --live-set=bounded"
# Second pass, plan 6.1.7: the zero-allocation decode invariant (and the
# rest of the suite) must hold under the SHIPPING configuration — release
# codegen with the bounded live-set discipline — not only the test profile.
"$CAJETA" --emit=exe --profile=test --release --live-set=bounded \
    --xpu-backend=cpu \
    --classpath="$out/llama.cja,$unit_cja,$codec_cja" \
    -o "$out/llamatests-release" \
    dev.cajeta.llama.selftest.TestMain.run "$here/src/test/cajeta" "$out" >/dev/null

"$out/llamatests-release"
