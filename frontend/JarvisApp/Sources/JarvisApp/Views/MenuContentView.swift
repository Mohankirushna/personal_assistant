import JarvisAppKit
import SwiftUI

struct MenuContentView: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Group {
            switch appState.status {
            case .connecting:
                Text("Connecting to backend…")
            case let .online(health):
                Text("Backend online (v\(health.version))")
                if let model = health.activeModel {
                    Text("Model: \(model)")
                } else {
                    Text(health.ollama.available ? "Model: idle" : "⚠️ Ollama unreachable")
                }
            case let .offline(reason):
                Text("Backend offline")
                Text(reason)
                    .font(.caption)
                Button("Retry") {
                    Task { await appState.start() }
                }
            }

            Divider()

            Button("Open Chat") {
                openWindow(id: "chat")
                NSApp.activate(ignoringOtherApps: true)
            }
            .keyboardShortcut("j")

            Button("Refresh Status") {
                Task { await appState.refreshHealth() }
            }

            Divider()

            Button("Quit Jarvis") {
                appState.shutdown()
                NSApp.terminate(nil)
            }
            .keyboardShortcut("q")
        }
    }
}
