import Foundation

/// HTTP + WebSocket client for the local Jarvis backend.
///
/// The backend only listens on loopback; `token` (when set) is sent as a
/// bearer token on every request except /health, matching the backend's
/// TokenAuthMiddleware.
public struct BackendClient {
    public let baseURL: URL
    public let token: String?
    private let session: URLSession

    public init(baseURL: URL = URL(string: "http://127.0.0.1:8765")!, token: String? = nil) {
        self.baseURL = baseURL
        self.token = token
        self.session = URLSession(configuration: .ephemeral)
    }

    public func authorizedRequest(path: String) -> URLRequest {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        if let token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    public func health() async throws -> HealthResponse {
        let request = authorizedRequest(path: "health")
        let (data, _) = try await session.data(for: request)
        return try JSONDecoder().decode(HealthResponse.self, from: data)
    }

    /// True if this client's credentials are accepted by an authenticated
    /// endpoint. /health cannot tell — it is deliberately auth-exempt, so a
    /// backend left over from a previous session looks healthy while
    /// rejecting every real call.
    public func canAuthenticate() async -> Bool {
        let request = authorizedRequest(path: "tools")
        guard
            let (_, response) = try? await session.data(for: request),
            let http = response as? HTTPURLResponse
        else { return false }
        return http.statusCode != 401 && http.statusCode != 403
    }

    public func chat(message: String, sessionId: String?) async throws -> ChatResponse {
        var request = authorizedRequest(path: "chat")
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "message": message,
            "session_id": sessionId as Any,
        ])
        let (data, response) = try await session.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw BackendError.http(status: http.statusCode, body: String(decoding: data, as: UTF8.self))
        }
        return try JSONDecoder().decode(ChatResponse.self, from: data)
    }

    /// Ask the backend to speak the morning briefing aloud, if audio is
    /// audible (it decides). Returns whether it spoke. Best-effort: a failure
    /// to reach the backend is swallowed by the caller.
    @discardableResult
    public func announceBriefing() async throws -> Bool {
        var request = authorizedRequest(path: "briefing/announce")
        request.httpMethod = "POST"
        let (data, response) = try await session.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw BackendError.http(status: http.statusCode, body: String(decoding: data, as: UTF8.self))
        }
        struct AnnounceResponse: Decodable { let spoken: Bool }
        return (try? JSONDecoder().decode(AnnounceResponse.self, from: data))?.spoken ?? false
    }

    /// Report the device's current city to the backend, for accurate local
    /// weather in the morning briefing. The backend subprocess can't obtain
    /// location itself (no app-bundle identity), so the app pushes it here.
    public func updateLocation(city: String) async throws {
        var request = authorizedRequest(path: "location")
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["city": city])
        let (data, response) = try await session.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw BackendError.http(status: http.statusCode, body: String(decoding: data, as: UTF8.self))
        }
    }

    /// Open /ws/chat, send one message, and stream events until "done"/"error".
    public func streamChat(
        message: String,
        sessionId: String?,
        confirmationHandler: @escaping @Sendable (ConfirmationRequest) async -> Bool
    ) -> AsyncThrowingStream<ChatStreamEvent, Error> {
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)!
        components.scheme = components.scheme == "https" ? "wss" : "ws"
        components.path = "/ws/chat"
        var request = URLRequest(url: components.url!)
        if let token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let task = session.webSocketTask(with: request)

        return AsyncThrowingStream { continuation in
            task.resume()
            Task {
                do {
                    let payload = try JSONSerialization.data(withJSONObject: [
                        "message": message,
                        "session_id": sessionId as Any,
                    ])
                    try await task.send(.string(String(decoding: payload, as: UTF8.self)))
                    receiveLoop: while true {
                        let raw = try await task.receive()
                        guard case let .string(text) = raw, let data = text.data(using: .utf8) else {
                            continue
                        }
                        let event = try JSONDecoder().decode(ChatStreamEvent.self, from: data)
                        continuation.yield(event)
                        switch event {
                        case let .confirmation(request):
                            let approved = await confirmationHandler(request)
                            let response = try JSONSerialization.data(withJSONObject: [
                                "type": "confirm_response",
                                "approved": approved,
                            ])
                            try await task.send(.string(String(decoding: response, as: UTF8.self)))
                            continue
                        case .done, .error: break receiveLoop
                        case .token: continue
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
                task.cancel(with: .normalClosure, reason: nil)
            }
            continuation.onTermination = { _ in
                task.cancel(with: .normalClosure, reason: nil)
            }
        }
    }
}

public enum BackendError: LocalizedError {
    case http(status: Int, body: String)

    public var errorDescription: String? {
        switch self {
        case let .http(status, body):
            return "Backend returned HTTP \(status): \(body)"
        }
    }
}
