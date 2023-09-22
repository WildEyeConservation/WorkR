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

library(overlap)
library(activity)
library(lubridate)
library(ggplot2)
library(solartime)

calculate_activity_pattern <- function(data, file_name, species, centre, unit, time, overlap, lat, lng, utc_offset_hours,tz) {
  # Calculate activity pattern
  dat <- data
  file_name <- paste0(file_name, ".JPG")
  jpeg(file = file_name, quality = 100, width = 800, height = 800, units = "px", pointsize = 16)

  max_result <- 0
  estimator <- "None"

  if (overlap == 'true' && length(species) == 2){
    dat1 <- dat[dat$species == species[1],]
    dat2 <- dat[dat$species == species[2],]

    if (time == "solar"){
      pos.time1 <- as.POSIXct(dat1$timestamp)
      time_solar1 <- solartime(pos.time1,  lat, lng, utc_offset_hours)
      time.rad1 <- time_solar1$solar

      pos.time2 <- as.POSIXct(dat2$timestamp)
      time_solar2 <- solartime(pos.time2,  lat, lng, utc_offset_hours)
      time.rad2 <- time_solar2$solar
    }
    else{
      pos.time1 <- as.POSIXct(dat1$timestamp)
      time.prop1 <- gettime(pos.time1, scale = "proportion")    
      time.rad1 <- gettime(pos.time1, scale = "radian")

      pos.time2 <- as.POSIXct(dat2$timestamp)
      time.prop2 <- gettime(pos.time2, scale = "proportion")    
      time.rad2 <- gettime(pos.time2, scale = "radian")
    }

    if (centre == 'night') {
      overlap_centre <- "midnight"
    }
    else{
      overlap_centre <- "noon"
    }

    line_types <- c(1, 2)

    if (length(time.rad1) > 50 && length(time.rad2) > 50){
      estimator = "Dhat4"
    }
    else{
      estimator = "Dhat1"
    }

    e = overlapEst(time.rad1, time.rad2, kmax = 3, adjust=c(0.8, 1, 4), n.grid = 128,
            type=estimator)

    overlapPlot(time.rad1, time.rad2, xscale = 24, xcenter = overlap_centre,
              linetype = c(1, 2), linecol = c("black", "black"), linewidth = c(1, 1),
              olapcol = "lightgrey", rug=FALSE, extend=NULL,
              n.grid = 128, kmax = 3, adjust = 1, main = paste("Overlap Estimate:", round(e, 2)))

    sunset = c()
    sunrise = c()
    for (i in seq_along(dat$timestamp)) {
      rise <- computeSunriseHour(as.Date(dat$timestamp[i]), lat, lng, utc_offset_hours)
      set <- computeSunsetHour(as.Date(dat$timestamp[i]), lat, lng, utc_offset_hours)

      sunrise <- append(sunrise, rise)
      sunset <- append(sunset, set)

    } 

    sunrise_avg <- mean(sunrise)
    sunset_avg <- mean(sunset)

    if (centre == 'night') {
      sunrise_avg <- sunrise_avg
      sunset_avg <- sunset_avg - 24
    }

    abline(v = sunrise_avg, col = "red", lty = 1)
    abline(v = sunset_avg, col = "red", lty = 1)

    legend("topright", legend = species, lty = line_types, cex = 0.8)
  }
  else{

    for (i in seq_along(species)) {
      s <- species[i]
      dat_s <- dat[dat$species == s,]

      if (time == "solar") {
        time_solar <- solartime(dat_s$timestamp, lat, lng, utc_offset_hours)
        time.rad <- time_solar$solar
      } else {
        pos.time <- as.POSIXct(dat_s$timestamp)
        time.prop <- gettime(pos.time, scale = "proportion")
        time.rad <- gettime(pos.time, scale = "radian")
      }

      result <- fitact(time.rad)

      if (i == 1) {
        plot(result, yunit = unit, data='none', centre = centre, col = 'black', tline = list(lty = i))
      } else {
        plot(result, yunit = unit, data='none', centre = centre, add = TRUE, col = 'black', tline = list(lty = i))
      }
      
    }

    sunset = c()
    sunrise = c()
    for (i in seq_along(dat$timestamp)) {
      rise <- computeSunriseHour(as.Date(dat$timestamp[i]), lat, lng, utc_offset_hours)
      set <- computeSunsetHour(as.Date(dat$timestamp[i]), lat, lng, utc_offset_hours)

      sunrise <- append(sunrise, rise)
      sunset <- append(sunset, set)

    } 

    sunrise_avg <- mean(sunrise)
    sunset_avg <- mean(sunset)

    if (centre == 'night') {
      sunrise_avg <- sunrise_avg
      sunset_avg <- sunset_avg - 24
    }

    abline(v = sunrise_avg, col = "red", lty = 1)
    abline(v = sunset_avg, col = "red", lty = 1)

    legend("topright", legend = species, col = "black", lty = seq_along(species), cex = 0.8)
  }

  dev.off()

  # Format sunrise and suset times
  sunrise_avg <- format(as.POSIXct(sunrise_avg, origin = "1970-01-01", tz = tz), "%H:%M")
  sunset_avg <- format(as.POSIXct(sunset_avg, origin = "1970-01-01", tz = tz), "%H:%M")
  
  return_list <- list('file_name' = file_name, 'estimator' = estimator, 'sunrise' = sunrise_avg, 'sunset' = sunset_avg)

  return (return_list)

}


get_activity_from_csv <- function(){
  # Update the parameters below for your use case

  # CSV file path
  filename <- 'data.csv'

  # Read csv file
  species_data <- read.csv(filename, header = TRUE, sep = ",")

  # Convert to R dataframe
  species_data <- as.data.frame(species_data)

  # Image filename to save (without extension)
  file_name <- 'activity_pattern'

  # Species 
  species <- c('Antelope', 'Leopard', 'Lion')

  # Centre (day or night)
  centre <- 'day'

  # Unit (density or frequency)
  unit <- 'density'

  # Time (solar or clock)
  time <- 'clock'

  # Overlap (true or false)
  overlap <- 'false'

  # Latitude
  lat <- -25.746

  # Longitude
  lng <- 28.187

  # UTC offset hours with sign (South Africa is UTC+2) 
  utc_offset_hours <- 2

  # Timezone
  tz <- 'Africa/Johannesburg'

  # Calculate activity pattern
  image_file_name <- calculate_activity_pattern(species_data, file_name, species, centre, unit, time, overlap, lat, lng, utc_offset_hours, tz)

  return(image_file_name)
}
