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

library(oSCR)
library(raster)
library(sf)
library(car)
library(carData)
library(Formula)
library(Rcapture)

trim <-0

spatial_capture_recapture <- function(edf, tdf, session_col, id_col, occ_col, trap_col, tag_col, sep, cov_names, cov_options, dh, file_names, shapefile_path, polygon_path, shx_path, zone_number, hemisphere){

    message = ''
    # 0. Rcapture
    # 0.1 Create capture history
    rownames(dh) <- dh$individual_id
    dh$individual_id <- NULL
    dhm <- as.matrix(dh)
    MRC0 <- closedp.0(dhm)

    cr <- as.data.frame(MRC0$results)
    # Add index column
    cr$Model <- rownames(cr)

    # remove rows where infofit is not 0
    cr <- cr[cr$infoFit == 0,]

    # remove infofit column
    cr$infoFit <- NULL

    # rename columns
    colnames(cr)[colnames(cr) == 'abundance'] <- 'Abundance'
    colnames(cr)[colnames(cr) == 'stderr'] <- 'Standard Error'
    colnames(cr)[colnames(cr) == 'deviance'] <- 'Deviance'
    colnames(cr)[colnames(cr) == 'df'] <- 'Degrees of Freedom'

    # reorder columns
    cr <- cr[,c(7,1,2,3,4,5,6)]

    # 1. Create oSCR data object
    cov_col_names <- c()
    if (length(cov_names) > 0){
        for (cov_name in cov_names){
            cov_col_names <- c(cov_col_names, cov_name)
        }
    }
    else{
        cov_col_names <- c()
    }

    for (cov_name in cov_col_names){
        if (cov_options[cov_name,'type'] == 'Numeric'){ # if numeric
            tdf[,cov_name] <- as.numeric(tdf[,cov_name])
            if (cov_options[cov_name, 'scale'] == 'Yes'){
                tdf[,cov_name] <- scale(tdf[,cov_name])
            }
        }
        else{ # if categorical
            tdf[,cov_name] <- as.factor(tdf[,cov_name])
        }
    }


    if (tag_col != "none"){
        if (length(cov_col_names) > 0){
            species.data <- 
                data2oscr(edf = edf, 
                            tdf = list(tdf), 
                            sess.col = which(colnames(edf) %in% session_col), 
                            id.col = which(colnames(edf) %in% id_col),
                            occ.col = which(colnames(edf) %in% occ_col), 
                            trap.col = which(colnames(edf) %in% trap_col),
                            sex.col = which(colnames(edf) %in% tag_col),
                            sex.nacode = "NA",
                            K = sum(grepl(occ_col, colnames(tdf))), 
                            ntraps = nrow(tdf),
                            trapcov.names = cov_col_names,
                            tdf.sep = sep)
        }
        else {
            species.data <- 
                data2oscr(edf = edf, 
                            tdf = list(tdf), 
                            sess.col = which(colnames(edf) %in% session_col), 
                            id.col = which(colnames(edf) %in% id_col),
                            occ.col = which(colnames(edf) %in% occ_col), 
                            trap.col = which(colnames(edf) %in% trap_col),
                            sex.col = which(colnames(edf) %in% tag_col),
                            sex.nacode = "NA",
                            K = sum(grepl(occ_col, colnames(tdf))), 
                            ntraps = nrow(tdf))
        }
    }
    else{
        if (length(cov_col_names) > 0){
            species.data <- 
                data2oscr(edf = edf, 
                            tdf = list(tdf), 
                            sess.col = which(colnames(edf) %in% session_col), 
                            id.col = which(colnames(edf) %in% id_col),
                            occ.col = which(colnames(edf) %in% occ_col), 
                            trap.col = which(colnames(edf) %in% trap_col),
                            K = sum(grepl(occ_col, colnames(tdf))), 
                            ntraps = nrow(tdf),
                            trapcov.names = cov_col_names,
                            tdf.sep = sep)
        }
        else {
            species.data <- 
                data2oscr(edf = edf, 
                            tdf = list(tdf), 
                            sess.col = which(colnames(edf) %in% session_col), 
                            id.col = which(colnames(edf) %in% id_col),
                            occ.col = which(colnames(edf) %in% occ_col), 
                            trap.col = which(colnames(edf) %in% trap_col),
                            K = sum(grepl(occ_col, colnames(tdf))), 
                            ntraps = nrow(tdf))
        }
    }


    # 2. Extract data from oSCR object
    species.sf <- species.data$scrFrame

    # get summary statistics
    nr_occasions <- species.sf$occasions
    mmdm <- species.sf$mmdm
    nr_sites <- nrow(tdf)
    nr_individuals <- length(unique(edf$individual_id))
    
    # Create summary df 
    summary_df <- data.frame(Individuals = nr_individuals, Sites = nr_sites, Occasions = nr_occasions, MMDM = mmdm, HMMDM = mmdm/2)

    if(mmdm == 0){
        message <- paste(message, 'The MMDM is 0. It can indicate that you have not specified coordinates for your sites. It can also indicate that no individuals were seen at more than one site (individual id may not be complete). Please ensure your data is correct and try again.')
        return (list(density = data.frame(), abundance = data.frame(), det_prob = data.frame(), sigma = data.frame(), summary = summary_df, aic = data.frame(), cr = cr, message = message, raster_df = data.frame(), sites_density = data.frame()))
    }
    hmmdm <- mmdm / 2
    

    # 3. Create state-space object
    buffer <- 3 * hmmdm
    resolution <- hmmdm / 2
    resolution <- round(resolution, 1)
    buffer <- round(buffer, 1)
    shapefile <- NULL
    if (shapefile_path == 'None' && polygon_path == 'None'){
        species.ss <- make.ssDF(species.sf, res=resolution, buff=buffer)
    }
    else{
        ss_buf <- make.ssDF(species.sf, res=resolution, buff=buffer)

        if (shapefile_path != 'None'){
            shapefile <- read_sf(shapefile_path)
        }
        else{
            # create shapefile from polygon geojson
            polygon <- st_read(polygon_path)
            file.remove(list.files(pattern="polygon"))
            st_write(polygon, "polygon.shp")
            shapefile <- read_sf("polygon.shp")
            if (hemisphere == 'N') {
                target_crs <- st_crs(paste0("+proj=utm +zone=", zone_number, " +datum=WGS84 +units=m +no_defs"))
            } else {
                target_crs <- st_crs(paste0("+proj=utm +zone=", zone_number, " +south +datum=WGS84 +units=m +no_defs"))
            }
            shapefile <- st_transform(shapefile, crs=target_crs)

        }

        cellsize <- resolution * 1000
        grd <- sf::st_make_grid(shapefile, cellsize=cellsize)
        grid <- (grd[shapefile])
        points <- sf::st_make_grid(shapefile, cellsize=cellsize, what="centers")
        points <- (points[grid])
        state_space <- st_coordinates(points)
        ss_df <- data.frame(X = state_space[,1]/1000, Y = state_space[,2]/1000, Tr = 1)
        species.ss <- list(ss_df)

        # Check if all sites are within the state-space
        for (i in 1:nrow(tdf)){
            site_x <- tdf[i,2]
            site_y <- tdf[i,3]
            distances <- sqrt((ss_df$X - site_x)^2 + (ss_df$Y - site_y)^2)
            min_index <- which.min(distances)
            min_distance <- distances[min_index]
            if (min_distance > resolution){
                message <- paste(message, 'One or more sites are outside the state-space. Please ensure that the state-space includes all sites or filter out sites that are outside the state-space and try again.')
                return (list(density = data.frame(), abundance = data.frame(), det_prob = data.frame(), sigma = data.frame(), summary = summary_df, aic = data.frame(), cr = cr, message = message, raster_df = data.frame(), sites_density = data.frame()))
            }
        }

        # Check if state-space is too large
        area_ss_buf <- resolution^2 * nrow(ss_buf[[1]])
        area_ss <- resolution^2 * nrow(ss_df)

        if (area_ss > area_ss_buf * 1.10){
            species.ss <- ss_buf
            shapefile <- NULL
            message <- paste(message, 'The masked state-space is too large (too much computation time). The unmasked state-space will be used instead.')
        }

    }

    # 4. Create oSCR model object
    t <- mmdm * 3
    # Always round up
    t <- ceiling(t)
    trim <<- t

    # null model
    m0 <- NULL
    tryCatch({
        m0 <- oSCR.fit(list(D~1,p0~1,sig~1), species.sf, species.ss, trimS=trim)
    }, error = function(e){
        print(e)
        message <- 'Model failed to fit. '
        m0 <- NULL
        print(message)
    })

    # Sex models
    sex_models <- list()
    if (tag_col != 'none'){
        tryCatch({
            ms1 <- oSCR.fit(list(D~1,p0~1,sig~sex), species.sf, species.ss, trimS=trim)
            # add to list
            sex_models[['ms1']] <- ms1
        }, error = function(e){
            print(e)
            message <- 'Model failed to fit. '
            ms1 <- NULL
        })

        tryCatch({
            ms2 <- oSCR.fit(list(D~1,p0~sex,sig~1), species.sf, species.ss, trimS=trim)
            sex_models[['ms2']] <- ms2
        }, error = function(e){
            print(e)
            message <- 'Model failed to fit. '
            ms2 <- NULL
        })

        tryCatch({
            ms3 <- oSCR.fit(list(D~1,p0~sex,sig~sex), species.sf, species.ss, trimS=trim)
            sex_models[['ms3']] <- ms3
        }, error = function(e){
            print(e)
            message <- 'Model failed to fit. '
            ms3 <- NULL
        })

    }

    cov_models <- list()
    if (length(cov_col_names) > 0){
        for (cov_name in cov_col_names){
            # p0
            tryCatch({
                cov_model_p <- paste(' ', cov_name, sep = "")
                cov_model_p <- paste("p0~", cov_model_p, sep = "")
                cov_model_p <- as.formula(cov_model_p)
                cov_model <- oSCR.fit(list(D~1,cov_model_p,sig~1), species.sf, species.ss, trimS=trim)
                cov_models[[cov_name]] <- cov_model
            }, error = function(e){
                print(e)
                message <- 'Model failed to fit. '
                cov_model <- NULL
            })
        }
    }

    # 5. Model selection 
    if (is.null(m0)){
        model_list <- list()
    }
    else{
        model_list <- list(m0=m0)
    }
    if (!is.null(sex_models)) {
        model_list <- c(model_list, sex_models)
    }
    if (!is.null(cov_models)) {
        model_list <- c(model_list, cov_models)
    }

    # Fit models
    if (length(model_list) == 0){
        message <- paste(message, 'No models were fitted. Please ensure that the data is correct and try again.')

        density <- data.frame()
        abundance <- data.frame()
        det_prob <- data.frame()
        sigma <- data.frame()
        aic_df <- data.frame()
        summary_df$best_model <- 'None'
        summary_df$best_model_formula <- 'None'
        raster_df <- data.frame()
        sites_density <- data.frame()
    }
    else{
        fl <- fitList.oSCR(model_list)
        # Model selection
        ms <- modSel.oSCR(fl)

        #AIC 
        aic <- ms$aic.tab
        aic_df <- as.data.frame(aic)

        # Add a column for model formulas to aic_df
        model_formulas <- c()
        for (i in 1:nrow(aic_df)){
            model_name <- aic_df[i,1]
            model <- model_list[[model_name]]$model
            model_formula <- paste(model[[1]], model[[2]], model[[3]], sep = " ")
            model_formulas <- c(model_formulas, model_formula)
        }
        aic_df$model_formula <- model_formulas

        # Rename columns in aic_df
        colnames(aic_df)[colnames(aic_df) == 'model'] <- 'Model'
        colnames(aic_df)[colnames(aic_df) == 'weight'] <- 'Weight'
        colnames(aic_df)[colnames(aic_df) == 'model_formula'] <- 'Model Formula'

        # Reorder columns in aic_df so that Model Formula is the second column
        aic_df <- aic_df[,c(1,8,2,3,4,5,6,7)]

        # Select best model
        model_name <- aic[1,1]
        best_model <- model_list[[model_name]]
        bm_model <- best_model$model
        best_model_formula <- paste(bm_model[[1]], bm_model[[2]], bm_model[[3]], sep = " ")
        summary_df$best_model <- model_name
        summary_df$best_model_formula <- best_model_formula

        # 6. Model predictions
        # 6.1 Density (res 100km2)
        factor <- 100/(resolution^2)
        pred.df.dens <- data.frame(Session = factor(1))
        pred.dens <- get.real(model = best_model, newdata = pred.df.dens, type = "dens", d.factor = factor)
        density <- as.data.frame(pred.dens)

        pred.abd <- get.real(model = best_model, newdata = pred.df.dens, type = "dens", d.factor = nrow(best_model$ssDF[[1]]))
        abundance <- as.data.frame(pred.abd)
        state_space <- nrow(best_model$ssDF[[1]])
        abundance$state_space <- state_space * (resolution^2)


        # 6.2 Encounter probability
        if (model_name == 'm0'){
            pred.df.det <- data.frame(Session = factor(1))
        }
        else if (tag_col != 'none' && model_name %in% c('ms1', 'ms2', 'ms3')){
            pred.df.det <- data.frame(Session = factor(1), sex=factor(c(0,1)))
        }
        else{
            model_data <- unique(tdf[,model_name])
            pred.df.det <- data.frame(Session = factor(1), model_name = model_data)
            colnames(pred.df.det)[colnames(pred.df.det) == 'model_name'] <- model_name
        }
        pred.det <- get.real(model = best_model, newdata = pred.df.det, type = "det")
        det_prob <- as.data.frame(pred.det)

        # 6.3 Sigma 
        if (model_name == 'm0'){
            pred.df.sigma <- data.frame(Session = factor(1))
        }
        else if (tag_col != 'none' && model_name %in% c('ms1', 'ms2', 'ms3')){
            pred.df.sigma <- data.frame(Session = factor(1), sex=factor(c(0,1)))
        }
        else{
            pred.df.sigma <- data.frame(Session = factor(1))
        }
        pred.sigma <- get.real(model = best_model, newdata = pred.df.sigma, type = "sig")
        sigma <- as.data.frame(pred.sigma)

        # 7. Plotting
        labs <- sapply(strsplit(as.character(tdf$site_id), "_"), "[", 1)
        # 7.1 Spatial Captures
        file_name <- paste0(file_names[1], ".JPG")
        jpeg(file = file_name, quality = 100, width = 800, height = 800, units = "px", pointsize = 16)
        plot(species.sf)
        title(xlab="X (UTM)", ylab="Y (UTM)")
        text(species.sf$traps[[1]], labels=labs, pos=3)
        dev.off()

        # 7.2 State-space
        if (shapefile != NULL){
            file_name <- paste0(file_names[2], ".JPG")
            jpeg(file = file_name, quality = 100, width = 800, height = 800, units = "px", pointsize = 16)
            plot(st_geometry(shapefile), lwd = 2, col = "white")
            plot(grid, add = TRUE) 
            plot(points, cex = 0.2, add=TRUE)
            traps <- species.sf$traps[[1]] * 1000
            points(traps, pch = 19)
            text(traps, labels = labs, pos = 3, offset = 0.5)
            dev.off()
        }
        else{
            file_name <- paste0(file_names[2], ".JPG")
            jpeg(file = file_name, quality = 100, width = 800, height = 800, units = "px", pointsize = 16)
            plot(species.ss, species.sf)
            text(species.sf$traps[[1]], labels=labs, pos=3)
            dev.off()
        }

        # 7.3 Density Map
        file_name <- paste0(file_names[3], ".JPG")
        jpeg(file = file_name, quality = 100, width = 800, height = 800, units = "px", pointsize = 16)
        pred <- predict.oSCR(scrFrame=species.sf, best_model, ssDF=species.ss)
        plot(pred$r[[1]], xlab="X (UTM)", ylab="Y (UTM)", zlab="Density")
        points(tdf[,2:3], pch=20, lwd=0.5)
        text(tdf[,2:3], labels=labs, pos=3)
        dev.off()

        # 8 Density map values
        # Get raster object and values and coordinates
        raster <- pred$r[[1]]
        r_ncols <- raster@ncols
        r_nrows <- raster@nrows
        r_extent <- raster@extent

        # Get coordinates
        raster_coordinates <- data.frame()
        for (i in 1:r_ncols){
            for (j in 1:r_nrows){
                x <- r_extent@xmin + (i * resolution)
                y <- r_extent@ymin + (j * resolution)
                raster_coordinates <- rbind(raster_coordinates, c(x,y))
            }
        }
        colnames(raster_coordinates) <- c('X', 'Y')

        raster_values <- extract(raster, raster_coordinates)
        raster_coordinates <- as.data.frame(raster_coordinates)
        raster_values <- as.data.frame(raster_values)

        colnames(raster_values) <- c('density')

        raster_df <- cbind(raster_coordinates, raster_values)

        # Replace NA with 0
        raster_df[is.na(raster_df)] <- 0

        # get density values for each site
        sites_density <- data.frame()
        for (i in 1:nrow(tdf)){
            site_id <- tdf[i,1]
            site_x <- tdf[i,2]
            site_y <- tdf[i,3]

            distances <- sqrt((raster_df$X - site_x)^2 + (raster_df$Y - site_y)^2)
            min_index <- which.min(distances)
            value <- raster_df[min_index, 3]
            # Get the closest density value
            # value <- raster_values[which.min(abs(raster_df$X - site_x) + abs(raster_df$Y - site_y)),1]
            if (is.na(value)){
                value <- 0.0
            }
            sites_density <- rbind(sites_density, c(site_id, value))
        }
        colnames(sites_density) <- c('site_id', 'density')
    }

    return (list(density = density, abundance = abundance, det_prob = det_prob, sigma = sigma, summary = summary_df, aic = aic_df, cr = cr, message = message, raster_df = raster_df, sites_density = sites_density))

}


get_scr <-function(){
    # Update parameters for own use case 

    # Get edf and tdf csv files
    edf <- read.csv("edf.csv", header = TRUE)
    tdf <- read.csv("tdf.csv", header = TRUE)
    dh <- read.csv("dh.csv", header = TRUE)

    # Get column names
    session_col <- "session"
    id_col <- "individual_id"
    occ_col <- "occasion"
    trap_col <- "site_id"
    # If sex is not available, set to "none"
    tag_col <- 'indiv_tags'

    # If covariates are not available, set to empty list
    cov_names <- c('Flash')
    sep <- '/'
    # Type can be 'Numeric' or 'Categorical' and scale can be 'Yes' or 'No'. The row name should be the name of the covariate.
    cov_options <- data.frame(type = c('Categorical'), scale = c('No'))
    rownames(cov_options) <- 'Flash'
    # Get file names for plots
    file_names <- c("Captures", "State_space", "Density_map")

    # Run function
    result <- spatial_capture_recapture(edf, tdf, session_col, id_col, occ_col, trap_col, tag_col, sep, cov_names, cov_options, dh, file_names)

    return (result)
}