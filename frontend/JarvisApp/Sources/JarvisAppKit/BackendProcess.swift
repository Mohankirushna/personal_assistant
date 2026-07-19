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
    private var stdinLeash: Pipe?
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
        // Development builds live at frontend/dist/Jarvis.app. Walk upward
        // rather than assuming a fixed parent count, so the app can find the
        // repository's sibling backend directory after it is bundled.
        var directory = Bundle.main.bundleURL.deletingLastPathComponent()
        for _ in 0..<5 {
            let candidate = directory.appendingPathComponent("backend", isDirectory: true)
            if FileManager.default.fileExists(
                atPath: candidate.appendingPathComponent("pyproject.toml").path
            ) {
                return candidate
            }
            directory.deleteLastPathComponent()
        }
        return nil
    }

    /// Finds uv without relying on the interactive-shell PATH.  launchd gives
    /// login items a deliberately small PATH, which normally excludes the
    /// per-user `~/.local/bin` installation used by uv.
    public static func resolveUVExecutable() -> URL? {
        let environment = ProcessInfo.processInfo.environment
        var candidates: [URL] = []

        if let configured = environment["JARVIS_UV_PATH"], !configured.isEmpty {
            candidates.append(URL(fileURLWithPath: configured))
        }
        candidates.append(
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".local/bin/uv")
        )
        candidates.append(URL(fileURLWithPath: "/opt/homebrew/bin/uv"))
        candidates.append(URL(fileURLWithPath: "/usr/local/bin/uv"))

        // Keep manually configured PATH installations working as a fallback.
        for directory in (environment["PATH"] ?? "").split(separator: ":") {
            candidates.append(URL(fileURLWithPath: String(directory)).appendingPathComponent("uv"))
        }

        return candidates.first { candidate in
            FileManager.default.isExecutableFile(atPath: candidate.path)
        }
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
            // /health alone is not enough: a backend leaked by a force-killed
            // app session still answers it but requires that session's token,
            // so every authenticated call (chat, /ws/voice) would fail.
            let client = BackendClient(baseURL: baseURL, token: nil)
            guard await client.canAuthenticate() else {
                throw SpawnError.staleBackendOnPort(port)
            }
            return client
        }
        guard let backendDir = Self.resolveBackendDirectory() else {
            throw SpawnError.backendDirectoryNotFound
        }
        guard let uvExecutable = Self.resolveUVExecutable() else {
            throw SpawnError.uvNotFound
        }
        let sessionToken = UUID().uuidString
        try launch(backendDir: backendDir, uvExecutable: uvExecutable, token: sessionToken)
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

    private func launch(backendDir: URL, uvExecutable: URL, token: String) throws {
        let process = Process()
        // Use an absolute executable path. A login item's launchd PATH is
        // usually only /usr/bin:/bin:/usr/sbin:/sbin.
        process.executableURL = uvExecutable
        process.arguments = ["run", "jarvis-backend"]
        process.currentDirectoryURL = backendDir
        var env = ProcessInfo.processInfo.environment
        env["JARVIS_AUTH_TOKEN"] = token
        env["JARVIS_PORT"] = String(port)
        // Leash the backend to this process: we hold the write end of its
        // stdin, and the backend exits on stdin EOF. If this app is
        // force-killed (launchctl bootout, crash) the pipe closes and the
        // backend follows, instead of leaking on the port with a token no
        // future session knows.
        env["JARVIS_EXIT_ON_STDIN_CLOSE"] = "1"
        let leash = Pipe()
        process.standardInput = leash
        process.environment = env
        try process.run()
        self.process = process
        self.stdinLeash = leash
    }

    public func stop() {
        process?.terminate()
        try? stdinLeash?.fileHandleForWriting.close()
        stdinLeash = nil
        process = nil
        token = nil
    }

    public enum SpawnError: LocalizedError {
        case backendDirectoryNotFound
        case uvNotFound
        case backendDidNotBecomeHealthy
        case staleBackendOnPort(Int)

        public var errorDescription: String? {
            switch self {
            case .backendDirectoryNotFound:
                return "Backend not running and no backend directory found. "
                    + "Start it manually (cd backend && uv run jarvis-backend) "
                    + "or set JARVIS_BACKEND_DIR."
            case .uvNotFound:
                return "The uv executable could not be found. Install uv or set JARVIS_UV_PATH "
                    + "to its absolute path."
            case .backendDidNotBecomeHealthy:
                return "Spawned the backend but /health never came up. "
                    + "Check that `uv run jarvis-backend` works in the backend directory."
            case let .staleBackendOnPort(port):
                return "A Jarvis backend from a previous session is still running on port "
                    + "\(port) and requires a token this session doesn't have. Quit it "
                    + "(pkill -f jarvis-backend) and Jarvis will start a fresh one."
            }
        }
    }
}
