import JarvisAppKit
import AppKit
import ServiceManagement
import SwiftUI

@main
struct JarvisApp: App {
    // MenuBarExtra content is built lazily (first click), so backend startup
    // must not live in the view tree — kick it off with the state object.
    @StateObject private var appState: AppState = {
        let state = AppState()
        state.observeWakeForBriefing()
        Task { await state.start() }
        return state
    }()

    init() {
        // Login is owned by scripts/install_login_item.sh.  Older builds also
        // registered through SMAppService, which could start a second copy of
        // the menu-bar app (and a second microphone stream) after login.
        try? SMAppService.mainApp.unregister()

        // A LaunchAgent executes the bundle binary directly, so opening the
        // .app afterwards can otherwise create another independent instance.
        // Keep the already-running instance and discard this duplicate before
        // it creates another backend/voice connection.
        let bundleID = Bundle.main.bundleIdentifier ?? "dev.jarvis.assistant"
        let currentPID = ProcessInfo.processInfo.processIdentifier
        if let running = NSRunningApplication
            .runningApplications(withBundleIdentifier: bundleID)
            .first(where: { $0.processIdentifier != currentPID }) {
            running.activate()
            DispatchQueue.main.async {
                NSApp.terminate(nil)
            }
        }
    }

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
