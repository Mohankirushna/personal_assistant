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

/// A safety-gated action that needs the user's approval before Jarvis can run it.
public struct ConfirmationRequest: Decodable, Equatable {
    public let tool: String
    public let risk: String
    public let action: String

    public init(tool: String, risk: String, action: String) {
        self.tool = tool
        self.risk = risk
        self.action = action
    }
}

/// Events on the /ws/chat stream.
public enum ChatStreamEvent: Equatable {
    case token(String)
    case confirmation(ConfirmationRequest)
    /// `speak`: true only when the user explicitly asked to have something
    /// read aloud (read_url_aloud ran this turn) — this text surface is
    /// otherwise silent by design. The caller should fetch /voice/speak for
    /// `reply` and play it when true.
    case done(sessionId: String, reply: String, speak: Bool)
    case error(String)
}

extension ChatStreamEvent: Decodable {
    private enum CodingKeys: String, CodingKey {
        case type, content, message, reply, tool, risk, action, speak
        case sessionId = "session_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(String.self, forKey: .type) {
        case "token":
            self = .token(try container.decode(String.self, forKey: .content))
        case "confirm_request":
            self = .confirmation(
                ConfirmationRequest(
                    tool: try container.decode(String.self, forKey: .tool),
                    risk: try container.decode(String.self, forKey: .risk),
                    action: try container.decode(String.self, forKey: .action)
                )
            )
        case "done":
            self = .done(
                sessionId: try container.decode(String.self, forKey: .sessionId),
                reply: try container.decode(String.self, forKey: .reply),
                speak: try container.decodeIfPresent(Bool.self, forKey: .speak) ?? false
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
