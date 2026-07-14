// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "JarvisApp",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "JarvisApp", targets: ["JarvisApp"]),
        .executable(name: "jarvis-app-selftest", targets: ["JarvisAppSelfTest"]),
        .library(name: "JarvisAppKit", targets: ["JarvisAppKit"]),
    ],
    targets: [
        // Non-UI logic: wire types, HTTP/WS client, backend process manager.
        .target(
            name: "JarvisAppKit",
            path: "Sources/JarvisAppKit",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // The SwiftUI menu-bar app.
        .executableTarget(
            name: "JarvisApp",
            dependencies: ["JarvisAppKit"],
            path: "Sources/JarvisApp",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // Assertion-based checks runnable without XCTest/swift-testing,
        // which Command Line Tools do not ship. See README.
        .executableTarget(
            name: "JarvisAppSelfTest",
            dependencies: ["JarvisAppKit"],
            path: "Sources/JarvisAppSelfTest",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
