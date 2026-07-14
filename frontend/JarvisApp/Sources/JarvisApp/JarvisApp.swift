import JarvisAppKit
import SwiftUI

@main
struct JarvisApp: App {
    // MenuBarExtra content is built lazily (first click), so backend startup
    // must not live in the view tree — kick it off with the state object.
    @StateObject private var appState: AppState = {
        let state = AppState()
        Task { await state.start() }
        return state
    }()

    var body: some Scene {
        MenuBarExtra("Jarvis", systemImage: menuBarIcon) {
            MenuContentView()
                .environmentObject(appState)
                .task { await appState.refreshHealth() }
        }

        Window("Jarvis Chat", id: "chat") {
            ChatView()
                .environmentObject(appState)
        }
        .defaultSize(width: 480, height: 560)
    }

    private var menuBarIcon: String {
        switch appState.status {
        case .connecting: "hourglass"
        case .online: "waveform.circle.fill"
        case .offline: "exclamationmark.triangle"
        }
    }
}
