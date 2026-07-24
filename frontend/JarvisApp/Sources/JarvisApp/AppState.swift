import AppKit
import AVFoundation
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
    @Published var pendingConfirmation: ConfirmationRequest?
    @Published var voiceStatus = "Voice: Off"
    @Published var voiceOverlayTranscript = ""
    @Published var voiceOverlayReply = ""
    @Published var voiceOverlayIsReplying = false

    private let processManager = BackendProcessManager()
    private var client: BackendClient?
    private lazy var locationProvider = LocationProvider { [weak self] in self?.client }
    private var sessionId: String?
    private var voiceAssistant: VoiceAssistant?
    // Independent of voiceAssistant, which may be nil (mic not yet started):
    // the text chat surface can speak an explicit "read this out loud" reply
    // even when voice mode was never activated this session.
    private var chatAudioPlayer: AVAudioPlayer?
    private let voiceOverlay = VoiceOverlayController()
    private var voiceOverlayHideTask: Task<Void, Never>?
    private var reconnectTask: Task<Void, Never>?
    private var healthMonitorTask: Task<Void, Never>?
    private var isStartingBackend = false
    private var wakeObserver: NSObjectProtocol?
    // The briefing speaks on system wake (lid open). It should also speak once
    // when the app itself launches/connects — login, reboot, or a manual
    // quit-and-reopen — which `didWakeNotification` never covers. This guard
    // keeps it to a single announcement per app run, so the health-check
    // reconnect loop can't re-trigger it.
    private var didAnnounceLaunchBriefing = false
    private enum ConfirmationSource { case chat, voice }
    private var confirmationSource: ConfirmationSource?

    /// Speak the morning briefing whenever the Mac wakes from sleep (lid
    /// opened, etc.). The backend decides whether audio is audible before
    /// speaking, so this just nudges it on every wake.
    func observeWakeForBriefing() {
        guard wakeObserver == nil else { return }
        wakeObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didWakeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, let client = self.client else { return }
                // Refresh the location so the briefing's weather matches where
                // the Mac woke up, then announce. Best-effort throughout:
                // never surface a wake-time network/location hiccup.
                self.locationProvider.requestLocation()
                _ = try? await client.announceBriefing()
            }
        }
    }

    /// Speak the briefing once when the app first connects at launch (login,
    /// reboot, or a manual quit-and-reopen). `didWakeNotification` only covers
    /// waking from sleep, so without this a fresh launch is silent. The backend
    /// still decides whether audio is audible (muted → it won't speak), and the
    /// guard means the reconnect loop can't announce it again.
    private func announceLaunchBriefingIfNeeded(_ client: BackendClient) {
        guard !didAnnounceLaunchBriefing else { return }
        didAnnounceLaunchBriefing = true
        Task { [weak self] in
            guard self != nil else { return }
            // A short beat lets the reported location land first, so the
            // briefing's weather matches where the Mac actually is.
            try? await Task.sleep(for: .seconds(2))
            _ = try? await client.announceBriefing()
        }
    }

    func start() async {
        // A health check and an automatic retry can arrive at the same time.
        // Keep a single launch attempt in flight so we never spawn competing
        // backends for the same port.
        guard !isStartingBackend else { return }
        isStartingBackend = true
        defer { isStartingBackend = false }

        status = .connecting
        do {
            let client = try await processManager.attachOrSpawn()
            self.client = client
            await refreshHealth()
            if status.isOnline {
                startVoice(with: client)
                startHealthMonitoring()
                // Report current location once connected, so the first
                // briefing already has accurate local weather.
                locationProvider.requestLocation()
                announceLaunchBriefingIfNeeded(client)
            }
        } catch {
            status = .offline(error.localizedDescription)
            startAutomaticReconnect()
        }
    }

    /// The menu-bar app can start before `uv` has finished warming its
    /// environment.  Keep retrying in the background so the user does not
    /// have to press Retry, and recover if a managed backend later exits.
    private func startAutomaticReconnect() {
        guard reconnectTask == nil else { return }
        reconnectTask = Task { [weak self] in
            defer { self?.reconnectTask = nil }
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(5))
                guard let self, !Task.isCancelled else { return }
                await self.start()
                if self.status.isOnline { return }
            }
        }
    }

    private func startHealthMonitoring() {
        guard healthMonitorTask == nil else { return }
        healthMonitorTask = Task { [weak self] in
            defer { self?.healthMonitorTask = nil }
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(10))
                guard let self, !Task.isCancelled else { return }
                await self.refreshHealth()
                if !self.status.isOnline { return }
            }
        }
    }

    private func startVoice(with client: BackendClient) {
        voiceAssistant?.stop()
        let voice = VoiceAssistant(client: client) { [weak self] event in
            self?.handleVoice(event)
        }
        voiceAssistant = voice
        Task {
            do {
                try await voice.start()
            } catch {
                voiceStatus = "Voice: \(error.localizedDescription)"
            }
        }
    }

    private func handleVoice(_ event: VoiceAssistant.Event) {
        switch event {
        case .waiting:
            voiceStatus = "Voice: Listening for “Hey Jarvis”"
        case .wakeDetected:
            voiceStatus = "Voice: Wake word heard"
            showVoiceOverlay(transcript: "I’m listening…", reply: "", isReplying: false)
        case .listening:
            voiceStatus = "Voice: Listening…"
            showVoiceOverlay(transcript: "I’m listening…", reply: "", isReplying: false)
        case let .transcript(text):
            voiceStatus = "Voice: Thinking…"
            if !text.isEmpty { messages.append(ChatMessage(role: .user, text: text)) }
            showVoiceOverlay(transcript: text, reply: "", isReplying: false)
        case let .toolActivity(tool, status):
            // Only "running" drives the UI; outcomes are covered by the
            // reply (or the confirmation flow for denials).
            guard status == "running" else { break }
            let phrase = ToolActivity.phrase(forTool: tool)
            voiceStatus = "Voice: \(phrase)…"
            showVoiceOverlay(
                transcript: voiceOverlayTranscript,
                reply: "\(phrase)…",
                isReplying: true
            )
        case let .reply(text):
            voiceStatus = "Voice: Listening for “Hey Jarvis”"
            if !text.isEmpty { messages.append(ChatMessage(role: .assistant, text: text)) }
            showVoiceOverlay(transcript: voiceOverlayTranscript, reply: text, isReplying: true)
            hideVoiceOverlay(after: .seconds(10))
        case let .confirmation(request):
            confirmationSource = .voice
            pendingConfirmation = request
            showVoiceOverlay(
                transcript: "Allow this action?",
                reply: request.action,
                isReplying: true
            )
        case .nothingHeard:
            voiceStatus = "Voice: Listening for “Hey Jarvis”"
            hideVoiceOverlay(after: .seconds(2))
        case let .error(message):
            voiceStatus = "Voice: \(message)"
            showVoiceOverlay(transcript: "Voice issue", reply: message, isReplying: true)
            hideVoiceOverlay(after: .seconds(8))
        }
    }

    private func showVoiceOverlay(transcript: String, reply: String, isReplying: Bool) {
        voiceOverlayHideTask?.cancel()
        voiceOverlayTranscript = transcript
        voiceOverlayReply = reply
        voiceOverlayIsReplying = isReplying
        voiceOverlay.show(using: self)
    }

    private func hideVoiceOverlay(after delay: Duration) {
        voiceOverlayHideTask?.cancel()
        voiceOverlayHideTask = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            self?.voiceOverlay.hide()
        }
    }

    /// Manual dismissal via the overlay's close button. Also the escape
    /// hatch when a stalled voice pipeline leaves the overlay showing with
    /// no event left to schedule an automatic hide.
    func dismissVoiceOverlay() {
        if pendingConfirmation != nil {
            resolveConfirmation(approved: false)
        } else {
            voiceOverlayHideTask?.cancel()
            voiceOverlay.hide()
        }
    }

    func refreshHealth() async {
        guard let client else { return }
        do {
            status = .online(try await client.health())
        } catch {
            status = .offline(error.localizedDescription)
            startAutomaticReconnect()
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
                for try await event in client.streamChat(
                    message: trimmed,
                    sessionId: sessionId,
                    confirmationHandler: { request in await self.confirm(request) }
                ) {
                    switch event {
                    case let .token(token):
                        appendToLastAssistantMessage(token)
                    case .confirmation:
                        // The confirmation dialog is managed by confirm(_:).
                        continue
                    case let .done(newSessionId, reply, speak):
                        sessionId = newSessionId
                        setLastAssistantMessage(reply)
                        if speak {
                            speakChatReply(reply)
                        }
                    case let .error(message):
                        setLastAssistantMessage("⚠️ \(message)")
                    }
                }
            } catch {
                setLastAssistantMessage("⚠️ \(error.localizedDescription)")
            }
        }
    }

    /// Speaks a chat reply the backend flagged as an explicit "read this out
    /// loud" request (see ChatStreamEvent.done). Best-effort: a fetch or
    /// playback failure here shouldn't disrupt the (already-shown) text reply.
    private func speakChatReply(_ text: String) {
        guard let client else { return }
        Task {
            do {
                let wav = try await client.speak(text: text)
                let player = try AVAudioPlayer(data: wav)
                chatAudioPlayer = player
                player.play()
            } catch {
                // Best-effort: the text reply is already shown, so a TTS
                // fetch/playback failure here isn't worth surfacing further.
            }
        }
    }

    private var confirmationContinuation: CheckedContinuation<Bool, Never>?

    private func confirm(_ request: ConfirmationRequest) async -> Bool {
        await withCheckedContinuation { continuation in
            confirmationSource = .chat
            confirmationContinuation = continuation
            pendingConfirmation = request
            showVoiceOverlay(
                transcript: "Allow this action?",
                reply: request.action,
                isReplying: true
            )
        }
    }

    func resolveConfirmation(approved: Bool) {
        guard let confirmationSource else { return }
        pendingConfirmation = nil
        switch confirmationSource {
        case .chat:
            confirmationContinuation?.resume(returning: approved)
        case .voice:
            voiceAssistant?.sendConfirmation(approved: approved)
        }
        confirmationContinuation = nil
        self.confirmationSource = nil
        voiceOverlayHideTask?.cancel()
        voiceOverlay.hide()
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
        voiceAssistant?.stop()
        voiceOverlayHideTask?.cancel()
        reconnectTask?.cancel()
        healthMonitorTask?.cancel()
        if let wakeObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(wakeObserver)
        }
        voiceOverlay.hide()
        processManager.stop()
    }
}
