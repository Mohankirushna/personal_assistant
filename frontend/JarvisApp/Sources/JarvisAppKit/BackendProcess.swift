import Foundation

/// Attaches to a running backend, or spawns one (`uv run jarvis-backend`)
/// with a per-session auth token, and terminates it when the app quits.
///
/// The backend directory is resolved from, in order:
///   1. the JARVIS_BACKEND_DIR environment variable,
///   2. the "backendDirectory" user default,
///   3. ../../backend relative to the app binary (the dev-tree layout).
public final class BackendProcessManager {
    private var process: Process?
    public let port: Int
    /// Token used for a backend we spawn; nil when attaching to an
    /// externally started backend (dev mode, no auth).
    private(set) var token: String?

    public init(port: Int = 8765) {
        self.port = port
    }

    public var baseURL: URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }

    public static func resolveBackendDirectory() -> URL? {
        let env = ProcessInfo.processInfo.environment
        if let fromEnv = env["JARVIS_BACKEND_DIR"] {
            return URL(fileURLWithPath: fromEnv, isDirectory: true)
        }
        if let fromDefaults = UserDefaults.standard.string(forKey: "backendDirectory") {
            return URL(fileURLWithPath: fromDefaults, isDirectory: true)
        }
        let devTree = Bundle.main.bundleURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("backend", isDirectory: true)
        if FileManager.default.fileExists(atPath: devTree.appendingPathComponent("pyproject.toml").path) {
            return devTree
        }
        return nil
    }

    /// True if a backend already answers /health on our port.
    public func backendIsRunning() async -> Bool {
        let client = BackendClient(baseURL: baseURL)
        return (try? await client.health()) != nil
    }

    /// Ensure a backend is available; returns a client configured for it.
    /// Throws if none is running and none could be spawned.
    public func attachOrSpawn() async throws -> BackendClient {
        if await backendIsRunning() {
            return BackendClient(baseURL: baseURL, token: nil)
        }
        guard let backendDir = Self.resolveBackendDirectory() else {
            throw SpawnError.backendDirectoryNotFound
        }
        let sessionToken = UUID().uuidString
        try launch(backendDir: backendDir, token: sessionToken)
        token = sessionToken

        // Wait for /health (model-free, so this is fast) up to 15s.
        let client = BackendClient(baseURL: baseURL, token: sessionToken)
        for _ in 0..<30 {
            if (try? await client.health()) != nil {
                return client
            }
            try await Task.sleep(for: .milliseconds(500))
        }
        stop()
        throw SpawnError.backendDidNotBecomeHealthy
    }

    private func launch(backendDir: URL, token: String) throws {
        let process = Process()
        // Login shell so uv (usually in ~/.local/bin) is on PATH.
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-lc", "exec uv run jarvis-backend"]
        process.currentDirectoryURL = backendDir
        var env = ProcessInfo.processInfo.environment
        env["JARVIS_AUTH_TOKEN"] = token
        env["JARVIS_PORT"] = String(port)
        process.environment = env
        try process.run()
        self.process = process
    }

    public func stop() {
        process?.terminate()
        process = nil
        token = nil
    }

    public enum SpawnError: LocalizedError {
        case backendDirectoryNotFound
        case backendDidNotBecomeHealthy

        public var errorDescription: String? {
            switch self {
            case .backendDirectoryNotFound:
                return "Backend not running and no backend directory found. "
                    + "Start it manually (cd backend && uv run jarvis-backend) "
                    + "or set JARVIS_BACKEND_DIR."
            case .backendDidNotBecomeHealthy:
                return "Spawned the backend but /health never came up. "
                    + "Check that `uv run jarvis-backend` works in the backend directory."
            }
        }
    }
}
