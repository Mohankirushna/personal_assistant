import Foundation

// Wire types mirroring the backend API (backend/app/api).

public struct OllamaStatus: Decodable, Equatable {
    public let available: Bool
    public let loadedModels: [String]

    public enum CodingKeys: String, CodingKey {
        case available
        case loadedModels = "loaded_models"
    }
}

public struct HealthResponse: Decodable, Equatable {
    public let status: String
    public let version: String
    public let ollama: OllamaStatus
    public let activeModel: String?

    public enum CodingKeys: String, CodingKey {
        case status, version, ollama
        case activeModel = "active_model"
    }
}

public struct ChatResponse: Decodable, Equatable {
    public let sessionId: String
    public let reply: String

    public enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case reply
    }
}

/// Events on the /ws/chat stream: {"type": "token" | "done" | "error", ...}
public enum ChatStreamEvent: Equatable {
    case token(String)
    case done(sessionId: String, reply: String)
    case error(String)
}

extension ChatStreamEvent: Decodable {
    private enum CodingKeys: String, CodingKey {
        case type, content, message, reply
        case sessionId = "session_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(String.self, forKey: .type) {
        case "token":
            self = .token(try container.decode(String.self, forKey: .content))
        case "done":
            self = .done(
                sessionId: try container.decode(String.self, forKey: .sessionId),
                reply: try container.decode(String.self, forKey: .reply)
            )
        case "error":
            self = .error(try container.decode(String.self, forKey: .message))
        case let other:
            throw DecodingError.dataCorruptedError(
                forKey: .type, in: container,
                debugDescription: "unknown event type '\(other)'"
            )
        }
    }
}

public struct ChatMessage: Identifiable, Equatable {
    public enum Role { case user, assistant }

    public let id = UUID()
    public let role: Role
    public var text: String

    public init(role: Role, text: String) {
        self.role = role
        self.text = text
    }
}
