# WorkR
# Copyright (C) 2023

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

library(Distance)

MIN_FOR_BINNING <- 25L
MIN_BINS <- 3L
MIN_PER_BIN <- 3L

.empty_result <- function() {
  list(
    status = "FAILURE",
    error = NULL,
    density = NA_real_,
    density_se = NA_real_,
    density_lci = NA_real_,
    density_uci = NA_real_,
    density_cv = NA_real_,
    density_df = NA_real_,
    n_detections = 0L,
    n_sites = 0L,
    model_key = NA_character_,
    model_name = NA_character_,
    sample_fraction = NA_real_,
    effective_detection_radius = NA_real_,
    left_trunc_effective = NA_real_,
    right_trunc_effective = NA_real_,
    used_cutpoints = FALSE,
    cutpoints = NA_character_,
    selection_method = NA_character_,
    chat = NA_real_,
    model_warnings = character(0),
    qaic_uniform = data.frame(),
    qaic_half_normal = data.frame(),
    qaic_hazard_rate = data.frame(),
    chi2_comparison = data.frame()
  )
}

.resolve_truncation <- function(distances, left_trunc, right_trunc) {
  left <- if (!is.na(left_trunc)) as.numeric(left_trunc) else 0
  right <- if (!is.na(right_trunc)) as.numeric(right_trunc) else max(distances, na.rm = TRUE)
  if (!is.finite(right)) {
    stop("Could not determine right truncation distance.")
  }
  if (right <= left) {
    stop("Right truncation must be greater than left truncation.")
  }
  list(left = left, right = right)
}

.auto_cutpoints <- function(left, right, distances, min_per_bin = MIN_PER_BIN, min_bins = MIN_BINS) {
  d <- distances[is.finite(distances) & distances >= left & distances <= right]
  if (length(d) < min_bins * min_per_bin) {
    return(NULL)
  }

  inner_end <- min(8, right)
  cps <- left
  if (inner_end > left) {
    fine <- seq(ceiling(left + 1e-9), floor(inner_end), by = 1)
    if (length(fine) > 0) {
      cps <- unique(c(cps, fine))
    }
  }
  if (right > max(cps)) {
    outer <- c(10, 12, 15, 18, 20, 25, 30)
    outer <- outer[outer > max(cps) & outer < right]
    cps <- sort(unique(c(cps, outer, right)))
  } else {
    cps <- sort(unique(c(cps, right)))
  }
  cps[1] <- left
  cps[length(cps)] <- right

  bin_counts <- function(cps, d) {
    counts <- numeric(length(cps) - 1L)
    for (i in seq_len(length(cps) - 1L)) {
      if (i == length(cps) - 1L) {
        counts[i] <- sum(d >= cps[i] & d <= cps[i + 1L])
      } else {
        counts[i] <- sum(d >= cps[i] & d < cps[i + 1L])
      }
    }
    counts
  }

  repeat {
    if (length(cps) <= min_bins + 1L) {
      break
    }
    counts <- bin_counts(cps, d)
    sparse <- which(counts < min_per_bin)
    if (length(sparse) == 0L) {
      break
    }
    idx <- sparse[1L]
    if (idx < length(cps) - 1L) {
      cps <- cps[-(idx + 1L)]
    } else if (idx > 1L) {
      cps <- cps[-idx]
    } else {
      break
    }
    cps[1] <- left
    cps[length(cps)] <- right
  }

  counts <- bin_counts(cps, d)
  if (sum(counts > 0L) < min_bins) {
    return(NULL)
  }
  cps
}

.try_fit_ds <- function(label, data, conversion, trunc_list, cutpoints = NULL, ...) {
  args <- list(
    data = data,
    transect = "point",
    convert_units = conversion,
    truncation = trunc_list,
    quiet = TRUE,
    ...
  )
  if (!is.null(cutpoints)) {
    args$cutpoints <- cutpoints
  }
  tryCatch({
    model <- do.call(ds, args)
    list(ok = TRUE, label = label, model = model, error = NA_character_)
  }, error = function(e) {
    list(ok = FALSE, label = label, model = NULL, error = conditionMessage(e))
  })
}

.fit_candidate_models <- function(data, conversion, trunc_list, cutpoints = NULL, small_n = FALSE) {
  if (small_n) {
    specs <- list(
      list(label = "uni1", key = "unif", adjustment = "cos", nadj = 1),
      list(label = "hn0", key = "hn", adjustment = NULL),
      list(label = "hr0", key = "hr", adjustment = NULL)
    )
  } else {
    specs <- list(
      list(label = "uni1", key = "unif", adjustment = "cos", nadj = 1),
      list(label = "uni2", key = "unif", adjustment = "cos", nadj = 2, monotonicity = "none"),
      list(label = "uni3", key = "unif", adjustment = "cos", nadj = 3, monotonicity = "none"),
      list(label = "hn0", key = "hn", adjustment = NULL),
      list(label = "hn1", key = "hn", adjustment = "cos", nadj = 1),
      list(label = "hn2", key = "hn", adjustment = "cos", nadj = 2),
      list(label = "hr0", key = "hr", adjustment = NULL),
      list(label = "hr1", key = "hr", adjustment = "poly", nadj = 1)
    )
  }

  fits <- lapply(specs, function(spec) {
    label <- spec$label
    fit_args <- spec[setdiff(names(spec), "label")]
    do.call(.try_fit_ds, c(
      list(label = label, data = data, conversion = conversion, trunc_list = trunc_list, cutpoints = cutpoints),
      fit_args
    ))
  })
  names(fits) <- vapply(specs, function(x) x$label, character(1))
  fits
}

.qaic_table <- function(fits) {
  ok <- fits[sapply(fits, function(x) x$ok)]
  if (length(ok) == 0L) {
    return(data.frame())
  }
  models <- lapply(ok, function(x) x$model)
  qaic <- do.call(QAIC, models)
  qaic <- as.data.frame(qaic)
  qaic$Model <- vapply(ok, function(x) x$label, character(1))
  if ("chat" %in% names(qaic)) {
    qaic$chat <- NULL
  }
  qaic[, c("Model", setdiff(names(qaic), "Model")), drop = FALSE]
}

.pick_qaic_winner <- function(fits) {
  ok <- fits[sapply(fits, function(x) x$ok)]
  if (length(ok) == 0L) {
    return(NULL)
  }
  if (length(ok) == 1L) {
    return(ok[[1L]])
  }
  qaic <- .qaic_table(ok)
  best_label <- qaic$Model[which.min(qaic$QAIC)]
  ok[[best_label]]
}

.select_detection_model <- function(fits) {
  warnings <- character(0)
  ok_fits <- fits[sapply(fits, function(x) x$ok)]
  if (length(ok_fits) == 0L) {
    stop("No detection function models converged.")
  }

  families <- list(
    uniform = c("uni1", "uni2", "uni3"),
    half_normal = c("hn0", "hn1", "hn2"),
    hazard_rate = c("hr0", "hr1")
  )

  family_winners <- list()
  qaic_uniform <- data.frame()
  qaic_half_normal <- data.frame()
  qaic_hazard_rate <- data.frame()

  for (family_name in names(families)) {
    family_fits <- fits[families[[family_name]]]
    family_fits <- family_fits[!sapply(family_fits, is.null)]
    family_fits <- family_fits[sapply(family_fits, function(x) is.list(x) && !is.null(x$label))]
    qaic_table <- .qaic_table(family_fits)
    if (family_name == "uniform") {
      qaic_uniform <- qaic_table
    } else if (family_name == "half_normal") {
      qaic_half_normal <- qaic_table
    } else {
      qaic_hazard_rate <- qaic_table
    }
    winner <- .pick_qaic_winner(family_fits)
    if (!is.null(winner)) {
      family_winners[[family_name]] <- winner
    } else {
      warnings <- c(warnings, paste("No converged model in", family_name, "family."))
    }
  }

  chi2_comparison <- data.frame()
  selection_method <- NA_character_
  selected <- NULL
  chat <- NA_real_

  if (length(family_winners) >= 2L) {
    winner_models <- lapply(family_winners, function(x) x$model)
    chi2_result <- do.call(chi2_select, winner_models)
    criteria <- as.numeric(chi2_result$criteria)
    modnames <- vapply(family_winners, function(x) {
      if (!is.null(x$model$ddf$name.message)) {
        as.character(x$model$ddf$name.message)
      } else {
        x$label
      }
    }, character(1))
    chi2_comparison <- data.frame(Model = modnames, chat = criteria, stringsAsFactors = FALSE)
    chi2_comparison <- chi2_comparison[order(chi2_comparison$chat), , drop = FALSE]
    best_idx <- which.min(criteria)
    winner_list <- unname(family_winners)
    selected <- winner_list[[best_idx]]
    selection_method <- "chi2_select"
    chat <- criteria[best_idx]
  } else if (length(family_winners) == 1L) {
    selected <- family_winners[[1L]]
    selection_method <- "single_family"
    if (!is.null(selected$model$ddf$fitted) && length(selected$model$ddf$fitted) > 0L) {
      chat <- NA_real_
    }
  } else {
    qaic_all <- .qaic_table(ok_fits)
    if (nrow(qaic_all) == 0L) {
      selected <- ok_fits[[1L]]
      selection_method <- "first_converged"
      warnings <- c(warnings, "Model selection fell back to the first converged model.")
    } else {
      best_label <- qaic_all$Model[which.min(qaic_all$QAIC)]
      selected <- ok_fits[[best_label]]
      selection_method <- "qaic_fallback"
      warnings <- c(warnings, "Model selection fell back to lowest QAIC across all converged models.")
    }
  }

  failed <- fits[sapply(fits, function(x) !x$ok)]
  if (length(failed) > 0L) {
    for (fit in failed) {
      warnings <- c(warnings, paste0(fit$label, " failed: ", fit$error))
    }
  }

  list(
    selected = selected,
    selection_method = selection_method,
    chat = chat,
    warnings = warnings,
    qaic_uniform = qaic_uniform,
    qaic_half_normal = qaic_half_normal,
    qaic_hazard_rate = qaic_hazard_rate,
    chi2_comparison = chi2_comparison
  )
}

.model_key_from_fit <- function(model) {
  name <- tryCatch(as.character(model$ddf$name.message), error = function(e) NA_character_)
  if (is.na(name)) {
    return(NA_character_)
  }
  if (grepl("half-normal", name, ignore.case = TRUE)) {
    return("half-normal")
  }
  if (grepl("hazard-rate", name, ignore.case = TRUE)) {
    return("hazard-rate")
  }
  if (grepl("uniform", name, ignore.case = TRUE)) {
    return("uniform")
  }
  name
}

.effective_detection_radius <- function(model, left, right) {
  if (is.null(model$ddf$fitted) || length(model$ddf$fitted) == 0L) {
    return(NA_real_)
  }
  p_a <- as.numeric(model$ddf$fitted[1L])
  w <- right - left
  sqrt(p_a * w^2)
}

calculate_distance_sampling <- function(flatfile, fov_degrees, left_trunc = NA, right_trunc = NA, plot_file = NULL) {
  # Camera-trap distance sampling with QAIC/chi2_select model selection (Howe et al. 2019).
  #
  # flatfile: Region.Label, Area, Sample.Label, Effort, distance, object
  # left_trunc: optional metres (default 0); right_trunc: optional (default max detection distance)
  # When n >= MIN_FOR_BINNING, distances are binned with auto cutpoints for model selection.

  result <- .empty_result()

  tryCatch({
    data <- as.data.frame(flatfile)

    required_cols <- c("Region.Label", "Area", "Sample.Label", "Effort", "distance")
    missing_cols <- required_cols[!(required_cols %in% names(data))]
    if (length(missing_cols) > 0) {
      stop(paste("Flatfile missing required columns:", paste(missing_cols, collapse = ", ")))
    }

    if (is.na(fov_degrees) || fov_degrees <= 0 || fov_degrees > 360) {
      stop("Field of view must be between 0 and 360 degrees.")
    }

    distances <- data$distance[!is.na(data$distance)]
    n_detections <- length(distances)
    n_sites <- length(unique(data$Sample.Label[!is.na(data$Sample.Label)]))

    if (n_detections < 2) {
      stop("Insufficient detections with distance values to fit detection function (need at least 2).")
    }

    if (!"object" %in% names(data) || all(is.na(data$object))) {
      data$object <- NA
      data$object[!is.na(data$distance)] <- seq_len(n_detections)
    }

    conversion <- convert_units("meter", NULL, "square kilometer")
    trunc <- .resolve_truncation(distances, left_trunc, right_trunc)
    trunc_list <- list(left = trunc$left, right = trunc$right)

    cutpoints <- NULL
    used_cutpoints <- FALSE
    small_n <- n_detections < MIN_FOR_BINNING
    if (!small_n) {
      cutpoints <- .auto_cutpoints(trunc$left, trunc$right, distances)
      used_cutpoints <- !is.null(cutpoints)
    }
    if (!used_cutpoints && n_detections >= MIN_FOR_BINNING) {
      small_n <- TRUE
    }

    fits <- .fit_candidate_models(data, conversion, trunc_list, cutpoints, small_n = small_n)
    selection <- .select_detection_model(fits)
    model <- selection$selected$model
    model_name <- tryCatch(as.character(model$ddf$name.message), error = function(e) selection$selected$label)
    model_key <- .model_key_from_fit(model)

    sample_fraction <- as.numeric(fov_degrees) / 360

    dens <- dht2(
      model,
      flatfile = data,
      strat_formula = ~1,
      stratification = "geographical",
      sample_fraction = sample_fraction,
      er_est = "P2",
      convert_units = conversion
    )

    drow <- attr(dens, "density")
    if (is.null(drow) || nrow(drow) < 1) {
      stop("Density estimation returned no results.")
    }

    pick_col <- function(candidates) {
      hit <- candidates[candidates %in% names(drow)]
      if (length(hit) == 0) {
        return(NA_real_)
      }
      as.numeric(drow[[hit[1]]][1])
    }

    result$density <- pick_col(c("Density", "Estimate"))
    result$density_se <- pick_col(c("Density_se", "se"))
    result$density_lci <- pick_col(c("LCI"))
    result$density_uci <- pick_col(c("UCI"))
    result$density_cv <- pick_col(c("Density_CV", "cv"))
    if ("df" %in% names(drow)) {
      result$density_df <- as.numeric(drow$df[1])
    }

    result$effective_detection_radius <- .effective_detection_radius(model, trunc$left, trunc$right)
    result$sample_fraction <- sample_fraction
    result$n_detections <- as.integer(n_detections)
    result$n_sites <- as.integer(n_sites)
    result$model_key <- model_key
    result$model_name <- model_name
    result$left_trunc_effective <- trunc$left
    result$right_trunc_effective <- trunc$right
    result$used_cutpoints <- used_cutpoints
    result$cutpoints <- if (used_cutpoints) paste(format(cutpoints, trim = TRUE, scientific = FALSE), collapse = ", ") else NA_character_
    result$selection_method <- selection$selection_method
    result$chat <- selection$chat
    result$model_warnings <- selection$warnings
    result$qaic_uniform <- selection$qaic_uniform
    result$qaic_half_normal <- selection$qaic_half_normal
    result$qaic_hazard_rate <- selection$qaic_hazard_rate
    result$chi2_comparison <- selection$chi2_comparison

    if (!is.null(plot_file) && !is.na(plot_file) && nzchar(as.character(plot_file))) {
      jpeg(
        filename = paste0(plot_file, ".JPG"),
        quality = 100,
        width = 800,
        height = 600,
        units = "px",
        pointsize = 16
      )
      plot(model, main = paste("Detection function:", model_name), xlab = "Distance (m)")
      dev.off()
    }

    result$status <- "SUCCESS"
    result$error <- NULL

  }, error = function(e) {
    result$status <<- "FAILURE"
    result$error <<- conditionMessage(e)
  })

  return(result)
}
