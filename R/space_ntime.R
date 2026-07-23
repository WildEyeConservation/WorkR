# WorkR
# Copyright (C) 2023
#
# spaceNtime time-to-event (TTE) abundance for camera traps.

library(spaceNtime)

.empty_tte_result <- function(error = NULL) {
  list(
    status = "FAILURE",
    error = error,
    N = NA_real_,
    SE = NA_real_,
    LCI = NA_real_,
    UCI = NA_real_,
    n_occasions = 0L,
    sampling_period_seconds = NA_real_
  )
}

calculate_space_ntime_tte <- function(
    df,
    deploy,
    study_start,
    study_end,
    species_speed_m_hr,
    study_area_m2,
    nper,
    time_btw_seconds
) {
  result <- .empty_tte_result()

    study_start <- as.POSIXct(study_start, tz = "UTC")
    study_end <- as.POSIXct(study_end, tz = "UTC")

    tryCatch({
    if (is.null(df) || nrow(df) < 1L) {
      stop("No detections found for the selected species and filters.")
    }
    if (is.null(deploy) || nrow(deploy) < 1L) {
      stop("No camera deployments found for the selected filters.")
    }
    if (is.na(study_area_m2) || study_area_m2 <= 0) {
      stop("Study area must be greater than 0 m².")
    }
    if (is.na(species_speed_m_hr) || species_speed_m_hr <= 0) {
      stop("Species speed must be greater than 0 m/hr.")
    }
    if (is.na(nper) || nper < 1L) {
      stop("Periods per occasion (nper) must be at least 1.")
    }
    if (is.na(time_btw_seconds) || time_btw_seconds < 0) {
      stop("Time between occasions must be non-negative.")
    }

    lps <- as.numeric(species_speed_m_hr) / 3600
    per <- tte_samp_per(deploy, lps = lps)
    if (!is.finite(per) || per <= 0) {
      stop("Could not derive a valid TTE sampling period from viewable area and species speed.")
    }

    occ <- tte_build_occ(
      per_length = per,
      nper = as.integer(nper),
      time_btw = as.numeric(time_btw_seconds),
      study_start = study_start,
      study_end = study_end
    )
    if (is.null(occ) || nrow(occ) < 1L) {
      stop("No sampling occasions could be built for the selected study dates and TTE settings.")
    }

    eh <- tte_build_eh(df, deploy, occ, samp_per = per, quiet = TRUE)
    est <- tte_estN_fn(eh, as.numeric(study_area_m2))

    result$status <- "SUCCESS"
    result$error <- NULL
    result$N <- as.numeric(est$N[1L])
    result$SE <- as.numeric(est$SE[1L])
    result$LCI <- as.numeric(est$LCI[1L])
    result$UCI <- as.numeric(est$UCI[1L])
    result$n_occasions <- as.integer(nrow(occ))
    result$sampling_period_seconds <- as.numeric(per)
  }, error = function(e) {
    result$error <- conditionMessage(e)
  })

  result
}
