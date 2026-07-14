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

    /// Open /ws/chat, send one message, and stream events until "done"/"error".
    public func streamChat(message: String, sessionId: String?) -> AsyncThrowingStream<ChatStreamEvent, Error> {
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
