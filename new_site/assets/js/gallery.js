(function () {
  "use strict";

  const root = document.getElementById("gallery-root");
  if (!root) return;

  fetch("data/gallery.json")
    .then(function (response) {
      if (!response.ok) throw new Error("Gallery data request failed");
      return response.json();
    })
    .then(renderGallery)
    .catch(function () {
      root.replaceChildren(createMessage(
        "The gallery could not be loaded. Please refresh the page to try again.",
        "gallery-error"
      ));
    });

  function renderGallery(courses) {
    if (!Array.isArray(courses) || courses.length === 0) {
      root.replaceChildren(createMessage("No gallery items have been added yet.", "gallery-error"));
      return;
    }

    const fragment = document.createDocumentFragment();
    fragment.appendChild(buildCourseNavigation(courses));

    courses.forEach(function (course) {
      fragment.appendChild(buildCourse(course));
    });

    root.replaceChildren(fragment);
    initializeLazyMedia(root);
  }

  function buildCourseNavigation(courses) {
    const nav = document.createElement("nav");
    nav.className = "gallery-course-nav";
    nav.setAttribute("aria-label", "Gallery collections");

    courses.forEach(function (course) {
      const link = document.createElement("a");
      link.href = "#gallery-" + course.id;
      link.textContent = course.title;
      nav.appendChild(link);
    });

    return nav;
  }

  function buildCourse(course) {
    const section = document.createElement("section");
    section.className = "gallery-course";
    section.id = "gallery-" + course.id;

    const details = document.createElement("details");
    details.className = "gallery-course-details";
    details.open = true;

    const summary = document.createElement("summary");
    summary.className = "gallery-course-header";

    const heading = document.createElement("h2");
    heading.textContent = course.title;

    summary.appendChild(heading);
    details.appendChild(summary);

    if (Array.isArray(course.items) && course.items.length > 0) {
      const grid = document.createElement("div");
      grid.className = "gallery-grid gallery-course-grid";

      course.items.forEach(function (item, itemIndex) {
        grid.appendChild(buildItem(
          course,
          item,
          "course",
          itemIndex,
          false
        ));
      });

      details.appendChild(grid);
    }

    if (Array.isArray(course.groups)) {
      course.groups.forEach(function (group, groupIndex) {
        details.appendChild(buildGroup(course, group, groupIndex));
      });
    }

    details.addEventListener("toggle", function () {
      if (!details.open) {
        details.querySelectorAll("video").forEach(function (video) {
          video.pause();
        });
      }
    });

    section.appendChild(details);
    return section;
  }

  function buildGroup(course, group, groupIndex) {
    const items = getItems(group);
    const details = document.createElement("details");
    details.className = "gallery-group";
    details.open = Boolean(group.open);

    const summary = document.createElement("summary");
    const title = document.createElement("span");
    title.className = "gallery-group-title";
    title.textContent = group.title;

    summary.appendChild(title);

    details.appendChild(summary);

    if (Array.isArray(group.examples)) {
      const suite = document.createElement("div");
      suite.className = "gallery-example-suite";

      group.examples.forEach(function (example, exampleIndex) {
        const exampleSet = buildExampleSet(course, example, groupIndex, exampleIndex);
        if (exampleSet) suite.appendChild(exampleSet);
      });

      if (suite.childElementCount > 0) details.appendChild(suite);
    }

    if (items.length > 0) {
      const grid = document.createElement("div");
      grid.className = "gallery-grid";
      if (Array.isArray(group.examples)) grid.classList.add("gallery-grid--trailing");

      items.forEach(function (item, itemIndex) {
        grid.appendChild(buildItem(
          course,
          item,
          String(groupIndex),
          itemIndex,
          Boolean(group.showLabels)
        ));
      });

      details.appendChild(grid);
    }

    details.addEventListener("toggle", function () {
      if (!details.open) {
        details.querySelectorAll("video").forEach(function (video) {
          video.pause();
        });
      }
    });

    return details;
  }

  function buildExampleSet(course, example, groupIndex, exampleIndex) {
    if (!Array.isArray(example.media) || example.media.length === 0) return null;

    const section = document.createElement("section");
    section.className = "gallery-example-set";
    if (example.label) section.setAttribute("aria-label", example.label);

    const grid = document.createElement("div");
    grid.className = "gallery-example-grid";

    example.media.forEach(function (item, itemIndex) {
      grid.appendChild(buildItem(
        course,
        item,
        groupIndex + "-" + exampleIndex,
        itemIndex,
        Boolean(example.label),
        example.label
      ));
    });

    section.appendChild(grid);
    return section;
  }

  function buildItem(course, item, itemGroupId, itemIndex, showLabel, visibleLabel) {
    const figure = document.createElement("figure");
    figure.className = "gallery-item" + (item.wide ? " gallery-item--wide" : "");

    const mediaFrame = document.createElement("div");
    mediaFrame.className = "gallery-media-frame gallery-media-frame--" + item.type;

    const mediaPath = "assets/gallery/" + course.id + "/" + item.file;
    const captionId = "gallery-caption-" + course.id + "-" + itemGroupId + "-" + itemIndex;
    const displayedLabel = visibleLabel || item.title || "";
    const accessibleLabel = item.accessibleLabel || item.title || displayedLabel || "Gallery visualization";

    let media;
    if (item.type === "video") {
      media = document.createElement("video");
      media.controls = true;
      media.loop = true;
      media.muted = true;
      media.defaultMuted = true;
      media.playsInline = true;
      media.preload = "none";
      media.dataset.src = mediaPath;
      media.setAttribute("aria-label", accessibleLabel);
    } else if (item.type === "interactive") {
      media = document.createElement("iframe");
      media.dataset.src = mediaPath;
      media.loading = "lazy";
      media.title = accessibleLabel;
    } else {
      media = document.createElement("img");
      media.dataset.src = mediaPath;
      media.alt = accessibleLabel;
      media.decoding = "async";
    }

    media.className = "gallery-media gallery-lazy-media";

    mediaFrame.appendChild(media);

    const hasVisibleCaption = (showLabel && displayedLabel) || item.caption || (item.creditUrl && item.creditLabel);
    if (hasVisibleCaption) {
      const figcaption = document.createElement("figcaption");
      figcaption.id = captionId;
      media.setAttribute("aria-describedby", captionId);

      if (showLabel && displayedLabel) {
        const title = document.createElement("h3");
        title.textContent = displayedLabel;
        figcaption.appendChild(title);
      }

      if (item.caption) {
        const caption = document.createElement("p");
        caption.textContent = item.caption;
        figcaption.appendChild(caption);
      }

      if (item.creditUrl && item.creditLabel) {
        const credit = document.createElement("a");
        credit.className = "gallery-credit";
        credit.href = item.creditUrl;
        credit.textContent = item.creditLabel;
        figcaption.appendChild(credit);
      }

      figure.append(mediaFrame, figcaption);
    } else {
      figure.classList.add("gallery-item--unlabeled");
      figure.appendChild(mediaFrame);
    }
    return figure;
  }

  function getItems(group) {
    if (Array.isArray(group.items)) return group.items;
    if (!group.series) return [];

    const items = [];
    for (let index = group.series.start; index <= group.series.end; index += 1) {
      const padded = String(index).padStart(2, "0");
      items.push({
        title: Array.isArray(group.series.titles)
          ? group.series.titles[index - group.series.start]
          : group.series.titlePrefix + " " + padded,
        file: group.series.directory + "/" + group.series.filePrefix + padded + group.series.fileSuffix,
        type: "video"
      });
    }
    return items;
  }

  function initializeLazyMedia(container) {
    const media = Array.from(container.querySelectorAll(".gallery-lazy-media"));
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const loadMedia = function (element) {
      if (!element.dataset.src) return;
      element.src = element.dataset.src;
      delete element.dataset.src;
      if (element.tagName === "VIDEO") element.load();
    };

    if (typeof window.IntersectionObserver === "undefined") {
      media.forEach(loadMedia);
      return;
    }

    const observer = new window.IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        const element = entry.target;
        if (entry.isIntersecting) {
          loadMedia(element);
          if (element.tagName === "VIDEO" && !reducedMotion) {
            const playPromise = element.play();
            if (playPromise) playPromise.catch(function () {});
          }
          if (element.tagName !== "VIDEO") observer.unobserve(element);
        } else if (element.tagName === "VIDEO") {
          element.pause();
        }
      });
    }, {
      rootMargin: "400px 0px",
      threshold: 0.05
    });

    media.forEach(function (element) {
      observer.observe(element);
    });
  }

  function createMessage(text, className) {
    const message = document.createElement("p");
    message.className = className;
    message.textContent = text;
    return message;
  }
})();
