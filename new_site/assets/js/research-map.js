(function () {
  "use strict";

  const mapElement = document.getElementById("research-map");
  const statusElement = document.getElementById("map-status");

  if (!mapElement) return;

  const setError = function (message) {
    mapElement.innerHTML = "";
    const error = document.createElement("p");
    error.className = "map-error";
    error.textContent = message;
    mapElement.appendChild(error);
  };

  if (typeof window.L === "undefined") {
    setError("The map library could not be loaded. Please check your connection and try again.");
    return;
  }

  fetch("data/research-locations.json")
    .then(function (response) {
      if (!response.ok) throw new Error("Location data request failed");
      return response.json();
    })
    .then(initializeMap)
    .catch(function () {
      setError("The research locations could not be loaded. Please refresh the page to try again.");
    });

  function initializeMap(locations) {
    if (!Array.isArray(locations) || locations.length === 0) {
      setError("No research locations have been added yet.");
      return;
    }

    mapElement.innerHTML = "";

    const sortedLocations = locations.slice().sort(function (a, b) {
      return b.year - a.year || a.city.localeCompare(b.city);
    });

    const map = window.L.map(mapElement, {
      center: [39, -45],
      zoom: 2,
      minZoom: 2,
      maxZoom: 8,
      dragging: true,
      scrollWheelZoom: true,
      doubleClickZoom: true,
      touchZoom: true,
      boxZoom: true,
      keyboard: true,
      zoomControl: true,
      worldCopyJump: true,
      fadeAnimation: false,
      zoomAnimation: false,
      markerZoomAnimation: false
    });

    window.L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    addResearchRoutes(map, sortedLocations);

    const academicJourneyIds = new Set(["columbus-oh", "hanover-nh", "dallas-tx"]);

    sortedLocations.forEach(function (location) {
      const markerIcon = window.L.divIcon({
        className: "research-marker-wrap" +
          (academicJourneyIds.has(location.id) ? " research-marker-wrap--academic" : ""),
        html: '<span class="research-marker" aria-hidden="true"></span>',
        iconSize: [18, 18],
        iconAnchor: [9, 9],
        popupAnchor: [0, -12],
        tooltipAnchor: [0, -11]
      });

      const marker = window.L.marker(location.coordinates, {
        icon: markerIcon,
        title: location.city + ", " + location.country,
        alt: "Show research details for " + location.city + ", " + location.country,
        keyboard: true,
        autoPanOnFocus: true,
        riseOnHover: true,
        zIndexOffset: location.id === "dallas-tx" ? 1000 : 0
      });

      marker.bindTooltip(buildTooltip(location), {
        className: "research-tooltip",
        direction: "top",
        opacity: 1,
        offset: [0, -2]
      });

      marker.bindPopup(buildPopup(location), {
        autoPan: true,
        closeButton: true,
        maxWidth: 280
      });

      marker.on("click", function () {
        announce("Selected " + location.city + ", " + location.country + ".");
      });

      marker.addTo(map);
    });

    const bounds = window.L.latLngBounds(sortedLocations.map(function (location) {
      return location.coordinates;
    }));

    const fitMap = function () {
      map.invalidateSize({ pan: false });
      map.fitBounds(bounds, {
        paddingTopLeft: [34, 34],
        paddingBottomRight: [34, 52],
        maxZoom: 3,
        animate: false
      });
    };

    window.requestAnimationFrame(fitMap);

    if (typeof window.ResizeObserver !== "undefined") {
      let resizeFrame = null;
      const observer = new window.ResizeObserver(function () {
        if (resizeFrame) window.cancelAnimationFrame(resizeFrame);
        resizeFrame = window.requestAnimationFrame(fitMap);
      });
      observer.observe(mapElement);
    } else {
      window.addEventListener("resize", fitMap);
    }

    mapElement.dataset.ready = "true";
    announce("Research map loaded with " + sortedLocations.length + " locations.");
  }

  function addResearchRoutes(map, locations) {
    const routePaneName = "research-routes";
    const routePane = map.createPane(routePaneName);
    routePane.style.zIndex = "425";
    routePane.style.pointerEvents = "none";

    const locationsById = new Map(locations.map(function (location) {
      return [location.id, location];
    }));

    const routes = [
      { from: "dallas-tx", to: "hanover-nh" },
      { from: "hanover-nh", to: "columbus-oh" }
    ];
    const arrows = [];

    routes.forEach(function (route) {
      const from = locationsById.get(route.from);
      const to = locationsById.get(route.to);
      if (!from || !to) return;

      const start = window.L.latLng(from.coordinates);
      const end = window.L.latLng(to.coordinates);

      window.L.polyline([start, end], {
        pane: routePaneName,
        color: "#171717",
        weight: 1.25,
        opacity: 0.85,
        lineCap: "round",
        lineJoin: "round",
        interactive: false
      }).addTo(map);

      [0.26, 0.42, 0.58, 0.74].forEach(function (fraction) {
        const startPoint = map.project(start);
        const endPoint = map.project(end);
        const position = map.unproject(
          window.L.point(
            startPoint.x + (endPoint.x - startPoint.x) * fraction,
            startPoint.y + (endPoint.y - startPoint.y) * fraction
          )
        );

        const arrow = window.L.marker(position, {
          pane: routePaneName,
          interactive: false,
          keyboard: false,
          icon: window.L.divIcon({
            className: "research-route-arrow-wrap",
            html: '<span class="research-route-arrow" aria-hidden="true"></span>',
            iconSize: [8, 8],
            iconAnchor: [4, 4]
          })
        }).addTo(map);

        arrows.push({ marker: arrow, start: start, end: end });
      });
    });

    const orientArrows = function () {
      arrows.forEach(function (arrow) {
        const startPoint = map.latLngToLayerPoint(arrow.start);
        const endPoint = map.latLngToLayerPoint(arrow.end);
        const angle = Math.atan2(
          endPoint.y - startPoint.y,
          endPoint.x - startPoint.x
        ) * 180 / Math.PI;
        const arrowElement = arrow.marker.getElement();
        const arrowHead = arrowElement && arrowElement.querySelector(".research-route-arrow");
        if (arrowHead) arrowHead.style.transform = "rotate(" + angle + "deg)";
      });
    };

    map.on("zoomend moveend viewreset", orientArrows);
    window.requestAnimationFrame(orientArrows);
  }

  function buildTooltip(location) {
    const tooltip = document.createElement("div");
    tooltip.className = "map-tooltip-card";

    const place = document.createElement("strong");
    place.className = "map-tooltip-place";
    place.textContent = location.city + ", " + displayRegion(location);

    const meta = document.createElement("span");
    meta.className = "map-tooltip-meta";
    meta.textContent = location.dateLabel + " / " + location.kind;

    const headline = document.createElement("span");
    headline.className = "map-tooltip-headline";
    headline.textContent = location.headline;

    tooltip.append(place, meta, headline);
    return tooltip;
  }

  function buildPopup(location) {
    const popup = document.createElement("article");

    const kicker = document.createElement("p");
    kicker.className = "map-popup-kicker";
    kicker.textContent = location.dateLabel + " / " + location.kind;

    const title = document.createElement("h3");
    title.className = "map-popup-title";
    title.textContent = location.city + ", " + displayRegion(location);

    const headline = document.createElement("p");
    headline.className = "map-popup-headline";
    headline.textContent = location.headline;

    popup.append(kicker, title, headline);
    return popup;
  }

  function displayRegion(location) {
    if (location.country === "United States") return location.region;
    return location.country;
  }

  function announce(message) {
    if (statusElement) statusElement.textContent = message;
  }
})();
