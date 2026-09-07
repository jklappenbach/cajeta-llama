#!/usr/bin/env bash
# Build + run the cajeta-llm unit tests.
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
# Run from the repo root: the fixtures (ChatTest.FIX, the tokenizer JSONs)
# are RELATIVE paths, and a launch from any other directory reads nil and
# null-derefs in the first test that opens one (measured 2026-09-07).
cd "$here"
CAJETA="${CAJETA:-cajeta}"


# --- artifact discovery -------------------------------------------------
# Where a checkout's .cja is. Prefers `cajeta artifact-path`, which reads
# that project's OWN manifest -- so a project that moves its artifacts with
# settings.output is followed rather than guessed, and the version comes
# from details.version instead of whichever file happens to be newest.
#
# Falls back to the historical build/archive glob only when the toolchain
# does not HAVE the verb (it lands after 0.24.0), so this keeps working on
# an older cajeta and starts using the verb as soon as a newer one is on
# PATH -- no flag day.
#
# The gate is the CAPABILITY, not the outcome. A fallback keyed on "the
# verb failed" would silently mask a verb that ran and answered wrongly,
# which is the very failure this replaces; keyed on "the verb is absent",
# it cannot. An empty result still means "not in this checkout", exactly
# as the glob did, so callers' registry fallbacks are unchanged.
cajeta_artifact_path() {
    local dir="$1" name="$2"
    local cj="${CAJETA:-${CAJETA_BIN:-cajeta}}"
    if [[ -z "${_cajeta_has_ap:-}" ]]; then
        if "$cj" artifact-path --help 2>/dev/null \
                | grep -q 'artifact-path \[options\]'; then
            _cajeta_has_ap=yes
        else
            _cajeta_has_ap=no
        fi
    fi
    if [[ "$_cajeta_has_ap" == yes ]]; then
        # Only report a path that EXISTS. The verb answers where the
        # artifact would be even when nothing has built it, but the glob
        # this replaces returned empty in that case, and every caller
        # reads empty as "not in this checkout" and falls back to the
        # registry. Handing back a path to a missing file instead would
        # turn that into a confusing compile failure.
        local p
        p=$( cd "$dir" 2>/dev/null && "$cj" artifact-path 2>/dev/null ) || return 0
        [[ -n "$p" && -f "$p" ]] && printf '%s\n' "$p"
        return 0
    else
        ls -t "$dir"/build/archive/"$name"-*.cja 2>/dev/null | head -1
    fi
}

# The COMPILE backend. CAJETA_XPU_BACKEND (the runtime dispatcher's env
# var) is accepted as a fallback spelling because a sweep invoked with
# only that var otherwise compiles for cpu and every device test
# early-returns — a vacuously green "amdgpu" suite, which is exactly
# what happened on 2026-08-23: the recorded 192/0/1 amdgpu sweep never
# ran a line of device code (proven by re-running both spellings on
# 2026-08-24: CAJETA_XPU_BACKEND alone reproduces 192/0/1 and the
# mma-tiering note names the cpu backend; XPU_BACKEND=amdgpu found 3
# failures the cpu compile could never see).
XPU_BACKEND="${XPU_BACKEND:-${CAJETA_XPU_BACKEND:-cpu}}"
echo ">> compile backend: ${XPU_BACKEND}"

# Run one suite binary and report what it ACTUALLY exercised.
#
# The pass count alone cannot tell a device run from a skipped one: a test
# that takes its `Device.activeBackend()` early-return still reports PASS,
# so cpu and amdgpu both print "257 passed" while 22 of those tests never
# touched a kernel on cpu (measured 2026-08-30). The compile-backend banner
# above catches the 2026-08-23 failure (compiled for cpu, labelled amdgpu);
# this catches the quieter one -- compiled and dispatched correctly, but the
# TESTS opted out. Print both skip families so a vacuous green is visible in
# the log instead of inferred later from a differential.
run_suite() {
    local bin="$1" label="$2" log
    log="$(mktemp)"
    set +e
    # Line-buffer stdout: the runtime prints its diagnostics (launch FAILED
    # etc.) to unbuffered stderr, and against block-buffered stdout they land
    # beside the WRONG test in the merged log — Unit 25 triage chased a
    # misattribution that pure buffering created. Line-buffered, adjacency
    # in the log is truth.
    stdbuf -oL -eL "$bin" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    local nocoop skipped
    nocoop=$(grep -c "has no coop kernel" "$log" || true)
    skipped=$(grep -c "xpu-kernel-skipped" "$log" || true)
    echo ">> ${label}: in-test device skips: ${nocoop} 'no coop kernel', ${skipped} 'xpu-kernel-skipped'"
    if [ "${nocoop}" != "0" ] || [ "${skipped}" != "0" ]; then
        echo ">> NOTE: ${nocoop} test(s) reported PASS without running device code."
        echo ">>       On a backend that claims device support this is a VACUOUS green."
    fi
    rm -f "$log"
    return $rc
}

# Ownership-migration switches (ownership/transfer-of-borrow compiler):
# the return-side (OWNED_BIND) and captured-borrow checks land warn-first
# there, and this library has NOT done its migration pass yet — the chat
# rewire fixed its own sites, but the tensor/model code has hundreds of
# owned-result receives. No-ops under released compilers that lack the
# checks. REMOVE both lines when llama's ownership migration closes.
export CAJETA_OWNED_BIND="${CAJETA_OWNED_BIND:-error}"
export CAJETA_CAPTURED_BORROW="${CAJETA_CAPTURED_BORROW:-error}"
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
    unit_cja="$(cajeta_artifact_path "$UNIT_REPO" dev.cajeta.unit 2>/dev/null)"
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
    codec_cja="$(cajeta_artifact_path "$CODEC_REPO" dev.cajeta.codec 2>/dev/null)"
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

# dev.cajeta.jinja (the chat-template engine, jinja plan Unit 9 /
# llama 13.18): sibling checkout first, then store, then a
# sha256-verified Olla fetch (once 9.2.1 publishes it).
JINJA_VER="$(sed -n 's/.*"dev\.cajeta\.jinja"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$here/cajeta.json" | head -1)"
JINJA_REPO="${JINJA_REPO:-$here/../cajeta-jinja}"
jinja_cja=""
if [[ -d "$JINJA_REPO" ]]; then
    echo ">> building dev.cajeta.jinja from checkout ($JINJA_REPO)"
    ( cd "$JINJA_REPO" && "$CAJETA" build >/dev/null )
    jinja_cja="$(cajeta_artifact_path "$JINJA_REPO" dev.cajeta.jinja 2>/dev/null)"
fi
if [[ -z "$jinja_cja" ]]; then
    jinja_cja="$OLLA_HOME/dev.cajeta.jinja/$JINJA_VER/dev.cajeta.jinja-$JINJA_VER.cja"
fi
if [[ ! -f "$jinja_cja" ]]; then
    jinja_cja="$here/build/.unit-cache/dev.cajeta.jinja-$JINJA_VER.cja"
    if [[ ! -f "$jinja_cja" ]]; then
        echo ">> fetching dev.cajeta.jinja $JINJA_VER from $OLLA_URL"
        meta="$(curl -fsS "$OLLA_URL/v2/resolve?name=dev.cajeta.jinja&version=$JINJA_VER")"
        sha="$(printf '%s' "$meta" | sed -n 's/.*"sha256":"sha256:\([0-9a-f]*\)".*/\1/p')"
        [[ -n "$sha" ]] || { echo "/v2/resolve gave no sha256 for jinja" >&2; exit 1; }
        mkdir -p "$(dirname "$jinja_cja")"
        curl -fsS -o "$jinja_cja" "$OLLA_URL/v2/blob/$sha"
        got="$(sha256_of "$jinja_cja")"
        [[ "$got" == "$sha" ]] || { rm -f "$jinja_cja"; echo "sha256 mismatch fetching jinja" >&2; exit 1; }
    fi
fi
echo ">> dev.cajeta.jinja: $jinja_cja"

# dev.cajeta.logging — the CLI's diagnostics backend (Julian, 2026-08-30).
# Same three-step resolution as codec/jinja: sibling checkout, then store,
# then a sha256-verified Olla fetch.
LOGGING_VER="$(sed -n 's/.*"dev\.cajeta\.logging"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$here/cajeta.json" | head -1)"
LOGGING_REPO="${LOGGING_REPO:-$here/../cajeta-logging}"
logging_cja=""
if [[ -d "$LOGGING_REPO" ]]; then
    echo ">> building dev.cajeta.logging from checkout ($LOGGING_REPO)"
    ( cd "$LOGGING_REPO" && "$CAJETA" build >/dev/null )
    logging_cja="$(cajeta_artifact_path "$LOGGING_REPO" dev.cajeta.logging 2>/dev/null)"
fi
if [[ -z "$logging_cja" ]]; then
    logging_cja="$OLLA_HOME/dev.cajeta.logging/$LOGGING_VER/dev.cajeta.logging-$LOGGING_VER.cja"
fi
if [[ ! -f "$logging_cja" ]]; then
    logging_cja="$here/build/.unit-cache/dev.cajeta.logging-$LOGGING_VER.cja"
    if [[ ! -f "$logging_cja" ]]; then
        echo ">> fetching dev.cajeta.logging $LOGGING_VER from $OLLA_URL"
        meta="$(curl -fsS "$OLLA_URL/v2/resolve?name=dev.cajeta.logging&version=$LOGGING_VER")"
        sha="$(printf '%s' "$meta" | sed -n 's/.*"sha256":"sha256:\([0-9a-f]*\)".*/\1/p')"
        [[ -n "$sha" ]] || { echo "/v2/resolve gave no sha256 for logging" >&2; exit 1; }
        mkdir -p "$(dirname "$logging_cja")"
        curl -fsS -o "$logging_cja" "$OLLA_URL/v2/blob/$sha"
        got="$(sha256_of "$logging_cja")"
        [[ "$got" == "$sha" ]] || { rm -f "$logging_cja"; echo "sha256 mismatch fetching logging" >&2; exit 1; }
    fi
fi
echo ">> dev.cajeta.logging: $logging_cja"

echo ">> building llama library .cja"
"$CAJETA" --emit=cja -o "$out/llama.cja" \
    --classpath="$codec_cja,$jinja_cja,$logging_cja" \
    dev.cajeta.llm.Llm.run "$here/src/main/cajeta" "$out" >/dev/null

echo ">> building + running the test binary"
# XPU_BACKEND (default cpu): the engine's device paths (device-resident weight
# loads, the decode kernels) are exercised on the portable CPU backend by
# default — the PlacementDispatchTests discipline, real KernelBuffers, no
# silicon needed, so the suite stays runnable anywhere.
#
# Override it to run the SAME suite on real silicon (plan 7.3.1's deferred
# follow-up, the Unit 15 gate's "runs on real silicon by definition"):
#
#   XPU_BACKEND=amdgpu,cpu ./run-tests.sh
#
# No coop-matrix override is needed: a kernel whose tiles straddle tiers (on
# amdgpu, f32 A/B operands are Portable while the f32 ACCUMULATOR is Native —
# it is the accumulator of f16/bf16 WMMA) now demotes to the portable tile as
# a GROUP and lowers, rather than being skipped. Fixed in cajeta 2026-08-21.
"$CAJETA" --emit=exe --profile=test --xpu-backend="${XPU_BACKEND:-cpu}" \
    --classpath="$out/llama.cja,$unit_cja,$codec_cja,$jinja_cja,$logging_cja" \
    -o "$out/llamatests" \
    dev.cajeta.llm.selftest.TestMain.run "$here/src/test/cajeta" "$out" >/dev/null

run_suite "$out/llamatests" "test profile"

echo ">> building + running the test binary under --release --live-set=bounded"
# Second pass, plan 6.1.7: the zero-allocation decode invariant (and the
# rest of the suite) must hold under the SHIPPING configuration — release
# codegen with the bounded live-set discipline — not only the test profile.
"$CAJETA" --emit=exe --profile=test --release --live-set=bounded \
    --xpu-backend="${XPU_BACKEND:-cpu}" \
    --classpath="$out/llama.cja,$unit_cja,$codec_cja,$jinja_cja,$logging_cja" \
    -o "$out/llamatests-release" \
    dev.cajeta.llm.selftest.TestMain.run "$here/src/test/cajeta" "$out" >/dev/null

run_suite "$out/llamatests-release" "release, bounded live-set"
