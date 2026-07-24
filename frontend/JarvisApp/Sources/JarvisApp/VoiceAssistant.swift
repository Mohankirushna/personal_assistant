import AVFoundation
import Foundation
import JarvisAppKit

/// Keeps a single always-on microphone stream connected to Jarvis's local
/// voice WebSocket. The backend performs wake-word detection, transcription,
/// planning, and speech synthesis; this class only captures and plays audio.
final class VoiceAssistant {
    enum Event {
        case waiting
        case wakeDetected
        case listening
        case transcript(String)
        case toolActivity(tool: String, status: String)
        case reply(String)
        case confirmation(ConfirmationRequest)
        case nothingHeard
        case error(String)
    }

    private let client: BackendClient
    private let onEvent: @MainActor (Event) -> Void
    private let session = URLSession(configuration: .default)
    private let capture = MicrophoneCapture()
    private var socket: URLSessionWebSocketTask?
    private var player: AVAudioPlayer?

    init(client: BackendClient, onEvent: @escaping @MainActor (Event) -> Void) {
        self.client = client
        self.onEvent = onEvent
    }

    func start() async throws {
        guard await requestMicrophoneAccess() else {
            await report(.error("Microphone access is required for Hey Jarvis."))
            return
        }
        connect()
        try capture.start { [weak self] pcm in
            self?.send(pcm)
        }
        await report(.waiting)
    }

    func stop() {
        capture.stop()
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
        player?.stop()
    }

    func sendConfirmation(approved: Bool) {
        guard let socket else { return }
        Task {
            do {
                let data = try JSONSerialization.data(withJSONObject: [
                    "type": "confirm_response",
                    "approved": approved,
                ])
                try await socket.send(.string(String(decoding: data, as: UTF8.self)))
            } catch {
                await report(.error("Could not send your approval: \(error.localizedDescription)"))
            }
        }
    }

    private func requestMicrophoneAccess() async -> Bool {
        await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    private func connect() {
        var components = URLComponents(url: client.baseURL, resolvingAgainstBaseURL: false)!
        components.scheme = components.scheme == "https" ? "wss" : "ws"
        components.path = "/ws/voice"
        var request = URLRequest(url: components.url!)
        if let token = client.token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let socket = session.webSocketTask(with: request)
        // URLSessionWebSocketTask defaults to a 1 MB max message size. Most
        // spoken replies are well under that, but reading a full article
        // aloud (read_url_aloud) can synthesize several minutes of audio —
        // a WAV well past 1 MB — sent as a single binary frame. Without
        // raising this, receive() throws past that point and the voice loop
        // dies silently (looks like "no audio played" with no error shown).
        socket.maximumMessageSize = 20 * 1024 * 1024
        self.socket = socket
        socket.resume()
        receiveNext(from: socket)
    }

    private func send(_ pcm: Data) {
        guard let socket else { return }
        Task {
            do {
                try await socket.send(.data(pcm))
            } catch {
                await report(.error("Voice connection lost: \(error.localizedDescription)"))
            }
        }
    }

    private func receiveNext(from socket: URLSessionWebSocketTask) {
        Task { [weak self] in
            guard let self else { return }
            do {
                let message = try await socket.receive()
                switch message {
                case let .string(text):
                    await handleEventJSON(text)
                case let .data(wav):
                    await play(wav: wav)
                @unknown default:
                    break
                }
                receiveNext(from: socket)
            } catch {
                await report(.error("Voice connection lost: \(error.localizedDescription)"))
            }
        }
    }

    private func handleEventJSON(_ text: String) async {
        guard
            let data = text.data(using: .utf8),
            let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let type = payload["type"] as? String
        else {
            await report(.error("Jarvis sent an invalid voice response."))
            return
        }
        switch type {
        case "wake": await report(.wakeDetected)
        case "listening": await report(.listening)
        case "transcript": await report(.transcript(payload["text"] as? String ?? ""))
        case "tool":
            await report(.toolActivity(
                tool: payload["tool"] as? String ?? "",
                status: payload["status"] as? String ?? ""
            ))
        case "reply": await report(.reply(payload["text"] as? String ?? ""))
        case "confirm_request":
            guard
                let tool = payload["tool"] as? String,
                let risk = payload["risk"] as? String,
                let action = payload["action"] as? String
            else {
                await report(.error("Jarvis sent an invalid confirmation request."))
                return
            }
            await report(.confirmation(ConfirmationRequest(tool: tool, risk: risk, action: action)))
        case "nothing_heard": await report(.nothingHeard)
        case "error": await report(.error(payload["message"] as? String ?? "Voice error"))
        default: break
        }
    }

    private func play(wav: Data) async {
        do {
            let player = try AVAudioPlayer(data: wav)
            self.player = player
            player.play()
        } catch {
            await report(.error("Could not play Jarvis's reply: \(error.localizedDescription)"))
        }
    }

    private func report(_ event: Event) async {
        await onEvent(event)
    }
}

private final class MicrophoneCapture {
    private let engine = AVAudioEngine()

    func start(onPCM: @escaping (Data) -> Void) throws {
        let input = engine.inputNode
        let inputFormat = input.inputFormat(forBus: 0)
        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: 16_000,
            channels: 1,
            interleaved: true
        ), let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw VoiceCaptureError.unsupportedFormat
        }

        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1_280, format: inputFormat) { buffer, _ in
            let capacity = AVAudioFrameCount(
                (Double(buffer.frameLength) * 16_000 / inputFormat.sampleRate).rounded(.up)
            )
            guard let converted = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else {
                return
            }
            var supplied = false
            var error: NSError?
            let status = converter.convert(to: converted, error: &error) { _, outStatus in
                if supplied {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                supplied = true
                outStatus.pointee = .haveData
                return buffer
            }
            guard status != .error, error == nil, let samples = converted.int16ChannelData else {
                return
            }
            let byteCount = Int(converted.frameLength) * MemoryLayout<Int16>.size
            onPCM(Data(bytes: samples[0], count: byteCount))
        }
        engine.prepare()
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }
}

private enum VoiceCaptureError: LocalizedError {
    case unsupportedFormat

    var errorDescription: String? {
        "This microphone format is not supported."
    }
}
