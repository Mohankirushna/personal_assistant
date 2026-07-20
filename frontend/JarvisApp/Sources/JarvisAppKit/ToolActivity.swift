import Foundation

/// Maps backend tool names (the `tool` field of `{"type": "tool"}` voice
/// events) to spoken-English activity lines for the overlay, e.g.
/// "Searching the web…" while `brave_search_open_first` runs.
public enum ToolActivity {
    private static let phrases: [String: String] = [
        "clock": "Checking the time",
        "battery_status": "Checking the battery",
        "list_running_apps": "Checking running apps",
        "list_bluetooth_devices": "Checking Bluetooth devices",
        "open_app": "Opening the app",
        "quit_app": "Quitting the app",
        "open_url": "Opening the website",
        "open_file": "Opening the file",
        "screenshot": "Taking a screenshot",
        "look_at_screen": "Looking at the screen",
        "media_control": "Controlling playback",
        "volume": "Adjusting the volume",
        "brightness": "Adjusting the brightness",
        "system_power": "Handling the power request",
        "window_arrange": "Arranging windows",
        "create_reminder": "Creating the reminder",
        "timer": "Setting the timer",
        "focus_mode": "Updating Focus",
        "calendar": "Checking your calendar",
        "clipboard_read": "Reading the clipboard",
        "clipboard_write": "Updating the clipboard",
        "terminal_run": "Running a command",
        "git": "Running git",
        "vscode_open": "Opening VS Code",
        "web_search": "Searching the web",
        "news_search": "Checking the news",
        "brave_search_open_first": "Searching the web",
        "browser_search": "Searching the web",
        "browser_open": "Opening the page",
        "browser_fill": "Filling in the page",
        "browser_download": "Downloading the file",
        "youtube_play": "Finding it on YouTube",
        "spotify_play": "Finding it on Spotify",
        "spotify_open_playlist": "Opening the playlist",
        "music_platform_prompt": "Checking music options",
        "whatsapp_send": "Sending the WhatsApp message",
    ]

    /// Unknown tools — including user plugins — fall back to the tool name
    /// with underscores converted to spaces.
    public static func phrase(forTool tool: String) -> String {
        if let phrase = phrases[tool] { return phrase }
        if tool.hasPrefix("finder_") { return "Working with your files" }
        let readable = tool.replacingOccurrences(of: "_", with: " ")
        return readable.isEmpty ? "Working on it" : "Running \(readable)"
    }
}
