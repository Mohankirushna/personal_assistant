import CoreLocation
import Foundation
import JarvisAppKit

/// Resolves the device's current city via Location Services and reports it to
/// the backend, so the morning briefing has accurate local weather.
///
/// This lives in the SwiftUI app on purpose: macOS Location Services only
/// grants (and only prompts) a process with an app-bundle identity, which the
/// Python backend subprocess doesn't have. The app requests the location and
/// pushes the resolved city to the backend's /location endpoint.
@MainActor
final class LocationProvider: NSObject, CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    private let geocoder = CLGeocoder()
    /// Read lazily so a backend that connects after launch is still picked up.
    private let clientProvider: () -> BackendClient?
    private var lastReportedCity: String?
    private var isGeocoding = false

    init(clientProvider: @escaping () -> BackendClient?) {
        self.clientProvider = clientProvider
        super.init()
        manager.delegate = self
        // City-level accuracy is all weather needs, and it's faster/cheaper
        // than precise positioning.
        manager.desiredAccuracy = kCLLocationAccuracyKilometer
    }

    /// Request permission (shows the prompt once) and a location fix. Safe to
    /// call repeatedly — e.g. again on wake to refresh before the briefing.
    func requestLocation() {
        if isAuthorized(manager.authorizationStatus) {
            manager.requestLocation()
        } else if manager.authorizationStatus == .notDetermined {
            manager.requestWhenInUseAuthorization()  // fix requested once authorized
        }
        // denied/restricted: briefing falls back to IP geolocation
    }

    /// macOS spells "when in use" authorization `.authorized` (the
    /// `.authorizedWhenInUse` case is iOS-only and unavailable here).
    private func isAuthorized(_ status: CLAuthorizationStatus) -> Bool {
        status == .authorized || status == .authorizedAlways
    }

    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            if self.isAuthorized(manager.authorizationStatus) {
                manager.requestLocation()
            }
        }
    }

    nonisolated func locationManager(
        _ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]
    ) {
        guard let location = locations.last else { return }
        Task { @MainActor in self.resolveCity(from: location) }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        // No fix available: leave the backend to fall back to IP geolocation.
    }

    private func resolveCity(from location: CLLocation) {
        guard !isGeocoding else { return }
        isGeocoding = true
        geocoder.reverseGeocodeLocation(location) { [weak self] placemarks, _ in
            Task { @MainActor in
                guard let self else { return }
                self.isGeocoding = false
                guard let city = placemarks?.first?.locality, !city.isEmpty else { return }
                self.report(city: city)
            }
        }
    }

    private func report(city: String) {
        guard city != lastReportedCity, let client = clientProvider() else { return }
        lastReportedCity = city
        Task { try? await client.updateLocation(city: city) }
    }
}
