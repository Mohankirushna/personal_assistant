// Self-test runner for wire-format decoding and client behavior.
//
// This exists because Command Line Tools (no full Xcode) ship neither XCTest
// nor swift-testing, and compiling swift-testing from source is unreasonable
// on 8GB hardware. Run with: swift run jarvis-app-selftest
// If Xcode is installed these checks can be promoted to a real test target.

import Foundation
import JarvisAppKit

var failures = 0

func expect(_ condition: Bool, _ label: String) {
    if condition {
        print("ok - \(label)")
    } else {
        failures += 1
        print("FAIL - \(label)")
    }
}

func expectThrows(_ label: String, _ body: () throws -> Void) {
    do {
        try body()
        failures += 1
        print("FAIL - \(label) (no error thrown)")
    } catch {
        print("ok - \(label)")
    }
}

// MARK: Health decoding
do {
    let json = """
    {"status": "ok", "version": "0.2.0",
     "ollama": {"available": true, "loaded_models": ["qwen2.5:3b-instruct-q4_K_M"]},
     "active_model": "qwen2.5:3b-instruct-q4_K_M"}
    """.data(using: .utf8)!
    let health = try JSONDecoder().decode(HealthResponse.self, from: json)
    expect(health.status == "ok", "health.status decodes")
    expect(health.ollama.available, "health.ollama.available decodes")
    expect(health.ollama.loadedModels == ["qwen2.5:3b-instruct-q4_K_M"], "loaded_models decodes")
    expect(health.activeModel == "qwen2.5:3b-instruct-q4_K_M", "active_model decodes")
} catch {
    failures += 1
    print("FAIL - health decoding threw: \(error)")
}

// MARK: Chat response decoding
do {
    let json = #"{"session_id": "abc123", "reply": "Hello."}"#.data(using: .utf8)!
    let response = try JSONDecoder().decode(ChatResponse.self, from: json)
    expect(response.sessionId == "abc123", "chat session_id decodes")
    expect(response.reply == "Hello.", "chat reply decodes")
} catch {
    failures += 1
    print("FAIL - chat decoding threw: \(error)")
}

// MARK: Stream events
do {
    let token = try JSONDecoder().decode(
        ChatStreamEvent.self, from: #"{"type": "token", "content": "Hi "}"#.data(using: .utf8)!)
    expect(token == .token("Hi "), "token event decodes")

    let done = try JSONDecoder().decode(
        ChatStreamEvent.self,
        from: #"{"type": "done", "session_id": "abc", "reply": "Hi there"}"#.data(using: .utf8)!)
    expect(done == .done(sessionId: "abc", reply: "Hi there"), "done event decodes")

    let errorEvent = try JSONDecoder().decode(
        ChatStreamEvent.self, from: #"{"type": "error", "message": "Ollama down"}"#.data(using: .utf8)!)
    expect(errorEvent == .error("Ollama down"), "error event decodes")

    let confirmation = try JSONDecoder().decode(
        ChatStreamEvent.self,
        from: #"{"type": "confirm_request", "tool": "browser_fill", "risk": "sensitive", "action": "browser_fill {…}"}"#.data(using: .utf8)!)
    expect(
        confirmation == .confirmation(
            ConfirmationRequest(tool: "browser_fill", risk: "sensitive", action: "browser_fill {…}")
        ),
        "confirmation event decodes"
    )
} catch {
    failures += 1
    print("FAIL - stream event decoding threw: \(error)")
}

expectThrows("unknown event type throws") {
    _ = try JSONDecoder().decode(
        ChatStreamEvent.self, from: #"{"type": "mystery"}"#.data(using: .utf8)!)
}

// MARK: Auth header
let authed = BackendClient(baseURL: URL(string: "http://127.0.0.1:8765")!, token: "tok123")
expect(
    authed.authorizedRequest(path: "chat").value(forHTTPHeaderField: "Authorization")
        == "Bearer tok123",
    "bearer token attached")

let anonymous = BackendClient(baseURL: URL(string: "http://127.0.0.1:8765")!)
expect(
    anonymous.authorizedRequest(path: "chat").value(forHTTPHeaderField: "Authorization") == nil,
    "no token means no header")

// MARK: Tool activity phrases
expect(
    ToolActivity.phrase(forTool: "brave_search_open_first") == "Searching the web",
    "known tool maps to a friendly phrase")
expect(
    ToolActivity.phrase(forTool: "finder_create_folder") == "Working with your files",
    "finder_* tools share the files phrase")
expect(
    ToolActivity.phrase(forTool: "my_custom_plugin") == "Running my custom plugin",
    "unknown tool falls back to a readable name")
expect(ToolActivity.phrase(forTool: "") == "Working on it", "empty tool name is handled")
expect(
    ToolActivity.phrase(forTool: "timer") == "Setting the timer",
    "timer has a dedicated phrase")
expect(
    ToolActivity.phrase(forTool: "calendar") == "Checking your calendar",
    "calendar has a dedicated phrase")

// MARK: Stale-backend detection
let staleError = BackendProcessManager.SpawnError.staleBackendOnPort(8765)
expect(
    staleError.errorDescription?.contains("8765") == true
        && staleError.errorDescription?.contains("previous session") == true,
    "stale backend error names the port and the cause")

// canAuthenticate must fail closed: no server listening -> not authenticated.
let unreachable = BackendClient(baseURL: URL(string: "http://127.0.0.1:59999")!)
let authResult = await unreachable.canAuthenticate()
expect(authResult == false, "canAuthenticate is false when nothing is listening")

if failures > 0 {
    print("\n\(failures) failure(s)")
    exit(1)
}
print("\nAll self-tests passed.")
