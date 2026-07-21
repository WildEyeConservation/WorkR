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

calculate_distance_sampling <- function(flatfile, fov_degrees, left_trunc = NA, right_trunc = NA, plot_file = NULL) {
  # Camera-trap distance sampling (CTDS) v1: half-normal detection function + density via dht2.
  #
  # flatfile: data.frame with columns Region.Label, Area, Sample.Label, Effort, distance, object
  # fov_degrees: camera field of view in degrees (sample_fraction = fov / 360)
  # left_trunc / right_trunc: optional truncation distances in metres (NA = none)
  # plot_file: optional path prefix (without extension) for detection function JPEG

  result <- list(
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
    model_key = "half-normal",
    sample_fraction = NA_real_,
    effective_detection_radius = NA_real_
  )

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

    n_detections <- sum(!is.na(data$distance))
    n_sites <- length(unique(data$Sample.Label[!is.na(data$Sample.Label)]))

    if (n_detections < 2) {
      stop("Insufficient detections with distance values to fit detection function (need at least 2).")
    }

    if (!"object" %in% names(data) || all(is.na(data$object))) {
      data$object <- NA
      data$object[!is.na(data$distance)] <- seq_len(n_detections)
    }

    conversion <- convert_units("meter", NULL, "square kilometer")

    trunc_list <- list()
    if (!is.na(left_trunc)) {
      trunc_list$left <- as.numeric(left_trunc)
    }
    if (!is.na(right_trunc)) {
      trunc_list$right <- as.numeric(right_trunc)
    }
    if (length(trunc_list) == 0) {
      trunc_list <- NULL
    }

    ds_args <- list(
      data = data,
      transect = "point",
      key = "hn",
      adjustment = NULL,
      convert_units = conversion
    )
    if (!is.null(trunc_list)) {
      ds_args$truncation <- trunc_list
    }

    model <- do.call(ds, ds_args)

    sample_fraction <- as.numeric(fov_degrees) / 360
    result$sample_fraction <- sample_fraction

    dens <- dht2(
      model,
      flatfile = data,
      strat_formula = ~1,
      sample_fraction = sample_fraction,
      er_est = "P2",
      convert_units = conversion
    )

    drow <- dens$individuals$D
    if (is.null(drow) || nrow(drow) < 1) {
      stop("Density estimation returned no results.")
    }

    result$density <- as.numeric(drow$Estimate[1])
    result$density_se <- as.numeric(drow$se[1])
    result$density_lci <- as.numeric(drow$LCI[1])
    result$density_uci <- as.numeric(drow$UCI[1])
    result$density_cv <- as.numeric(drow$cv[1])
    if ("df" %in% names(drow)) {
      result$density_df <- as.numeric(drow$df[1])
    }

    # Effective detection radius from fitted average detection probability (if available)
    if (!is.null(model$ddf$fitted) && length(model$ddf$fitted) > 0 && !is.null(trunc_list$right)) {
      p_a <- as.numeric(model$ddf$fitted[1])
      w <- trunc_list$right - if (!is.null(trunc_list$left)) trunc_list$left else 0
      result$effective_detection_radius <- sqrt(p_a * w^2)
    }

    if (!is.null(plot_file) && !is.na(plot_file) && nzchar(as.character(plot_file))) {
      jpeg(
        filename = paste0(plot_file, ".JPG"),
        quality = 100,
        width = 800,
        height = 600,
        units = "px",
        pointsize = 16
      )
      plot(model, main = "Detection function (half-normal)", xlab = "Distance (m)")
      dev.off()
    }

    result$n_detections <- as.integer(n_detections)
    result$n_sites <- as.integer(n_sites)
    result$status <- "SUCCESS"
    result$error <- NULL

  }, error = function(e) {
    result$status <<- "FAILURE"
    result$error <<- conditionMessage(e)
  })

  return(result)
}
