import Foundation
import JarvisAppKit
import SwiftUI

@MainActor
final class AppState: ObservableObject {
    enum BackendStatus: Equatable {
        case connecting
        case online(HealthResponse)
        case offline(String)

        var isOnline: Bool {
            if case .online = self { return true }
            return false
        }
    }

    @Published var status: BackendStatus = .connecting
    @Published var messages: [ChatMessage] = []
    @Published var isReplying = false

    private let processManager = BackendProcessManager()
    private var client: BackendClient?
    private var sessionId: String?

    func start() async {
        status = .connecting
        do {
            let client = try await processManager.attachOrSpawn()
            self.client = client
            await refreshHealth()
        } catch {
            status = .offline(error.localizedDescription)
        }
    }

    func refreshHealth() async {
        guard let client else { return }
        do {
            status = .online(try await client.health())
        } catch {
            status = .offline(error.localizedDescription)
        }
    }

    func send(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let client, !isReplying else { return }
        messages.append(ChatMessage(role: .user, text: trimmed))
        messages.append(ChatMessage(role: .assistant, text: ""))
        isReplying = true

        Task {
            defer { isReplying = false }
            do {
                for try await event in client.streamChat(message: trimmed, sessionId: sessionId) {
                    switch event {
                    case let .token(token):
                        appendToLastAssistantMessage(token)
                    case let .done(newSessionId, reply):
                        sessionId = newSessionId
                        setLastAssistantMessage(reply)
                    case let .error(message):
                        setLastAssistantMessage("⚠️ \(message)")
                    }
                }
            } catch {
                setLastAssistantMessage("⚠️ \(error.localizedDescription)")
            }
        }
    }

    private func appendToLastAssistantMessage(_ token: String) {
        guard let index = messages.lastIndex(where: { $0.role == .assistant }) else { return }
        messages[index].text += token
    }

    private func setLastAssistantMessage(_ text: String) {
        guard let index = messages.lastIndex(where: { $0.role == .assistant }) else { return }
        messages[index].text = text
    }

    func shutdown() {
        processManager.stop()
    }
}
