import JarvisAppKit
import SwiftUI

struct ChatView: View {
    @EnvironmentObject var appState: AppState
    @State private var draft = ""

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 10) {
                        ForEach(appState.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: appState.messages) {
                    if let last = appState.messages.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
            Divider()
            HStack {
                TextField("Ask Jarvis…", text: $draft)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(sendDraft)
                Button("Send", action: sendDraft)
                    .keyboardShortcut(.defaultAction)
                    .disabled(appState.isReplying || !appState.status.isOnline)
            }
            .padding(10)
        }
        .frame(minWidth: 420, minHeight: 480)
    }

    private func sendDraft() {
        appState.send(draft)
        draft = ""
    }
}

private struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 40) }
            Text(message.text.isEmpty ? "…" : message.text)
                .textSelection(.enabled)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    message.role == .user
                        ? Color.accentColor.opacity(0.85)
                        : Color(nsColor: .controlBackgroundColor)
                )
                .foregroundStyle(message.role == .user ? .white : .primary)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            if message.role == .assistant { Spacer(minLength: 40) }
        }
    }
}
