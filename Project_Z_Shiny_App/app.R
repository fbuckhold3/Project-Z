# Project Z - ABC Schedule Explorer
# Interactive Shiny tool for residency schedule modeling
# Run with: shiny::runApp("Project_Z_Shiny_App")

library(shiny)
library(DT)

# ═══════════════════════════════════════════════════════════════════
# SCHEDULING ENGINE
# ═══════════════════════════════════════════════════════════════════

build_schedule <- function(n_pgy3 = 27, n_pgy2 = 28,
                           n_sluh = 6, n_va = 5, n_micu = 4, n_nf = 5,
                           n_bronze = 2, n_cards = 2, n_diamond = 1,
                           n_gold = 1, n_id = 1, n_jeopardy = 1,
                           seed = 42) {

  set.seed(seed)

  N_BLOCKS <- 8
  BW <- 6
  TW <- 48

  n_upper <- n_pgy3 + n_pgy2
  pgy_vec <- c(rep(3, n_pgy3), rep(2, n_pgy2))
  ids <- c(paste0("y3_", seq_len(n_pgy3)), paste0("y2_", seq_len(n_pgy2)))

  # Clinic positions 1-6
  base_per_pos <- n_upper %/% 6
  extra <- n_upper %% 6
  pos_counts <- rep(base_per_pos, 6)
  if (extra > 0) pos_counts[1:extra] <- pos_counts[1:extra] + 1
  cpos <- sample(rep(1:6, pos_counts))

  # Schedule matrix
  sched <- matrix("", nrow = n_upper, ncol = TW)
  rownames(sched) <- ids

  for (i in seq_len(n_upper)) {
    for (b in 0:(N_BLOCKS - 1)) {
      sched[i, b * BW + cpos[i]] <- "Clinic"
    }
  }

  u_rots <- list(
    rSLUH = n_sluh, rVA = n_va, ID = n_id, rNF = n_nf,
    rMICU = n_micu, Bronze = n_bronze, Cards = n_cards,
    Diamond = n_diamond, Gold = n_gold, Jeopardy = n_jeopardy
  )

  annual <- matrix(0, nrow = n_upper, ncol = length(u_rots))
  colnames(annual) <- names(u_rots)
  rownames(annual) <- ids

  get_nonclinic <- function(cp) setdiff(1:6, cp)

  for (block in 0:(N_BLOCKS - 1)) {
    sw <- block * BW
    used <- logical(n_upper)

    # 3-week rotations
    three_wk <- list(
      list(rot = "rSLUH", need = n_sluh, target = 1:3),
      list(rot = "rSLUH", need = n_sluh, target = 4:6),
      list(rot = "rVA", need = n_va, target = 1:3),
      list(rot = "rVA", need = n_va, target = 4:6),
      list(rot = "ID", need = n_id, target = 1:3),
      list(rot = "ID", need = n_id, target = 4:6)
    )

    for (tw in three_wk) {
      eligible <- which(!used & sapply(seq_len(n_upper), function(i) {
        all(tw$target %in% get_nonclinic(cpos[i]))
      }))
      if (length(eligible) == 0) next

      scores <- annual[eligible, tw$rot]
      eligible <- eligible[order(scores, runif(length(eligible)))]

      assigned <- 0
      for (idx in eligible) {
        if (assigned >= tw$need) break
        sched[idx, sw + tw$target] <- tw$rot
        ncw <- get_nonclinic(cpos[idx])
        op_weeks <- setdiff(ncw, tw$target)
        sched[idx, sw + op_weeks] <- "OP"
        used[idx] <- TRUE
        annual[idx, tw$rot] <- annual[idx, tw$rot] + 1
        assigned <- assigned + 1
      }
    }

    # 2-week NF
    for (cohort in list(1:2, 3:4, 5:6)) {
      eligible <- which(!used & sapply(seq_len(n_upper), function(i) {
        all(cohort %in% get_nonclinic(cpos[i]))
      }))
      if (length(eligible) == 0) next

      scores <- annual[eligible, "rNF"]
      eligible <- eligible[order(scores, runif(length(eligible)))]

      assigned <- 0
      for (idx in eligible) {
        if (assigned >= n_nf) break
        sched[idx, sw + cohort] <- "rNF"
        ncw <- get_nonclinic(cpos[idx])
        remaining <- setdiff(ncw, cohort)
        if (length(remaining) >= 3) {
          sched[idx, sw + remaining[2]] <- "IP_SINGLE"
          sched[idx, sw + remaining[1]] <- "OP"
          sched[idx, sw + remaining[3]] <- "OP"
        }
        used[idx] <- TRUE
        annual[idx, "rNF"] <- annual[idx, "rNF"] + 1
        assigned <- assigned + 1
      }
    }

    # Remaining: singles
    for (idx in which(!used)) {
      ncw <- get_nonclinic(cpos[idx])
      empty <- ncw[sched[idx, sw + ncw] == ""]
      if (length(empty) >= 5) {
        sched[idx, sw + empty[1:3]] <- "IP_SINGLE"
        sched[idx, sw + empty[4:5]] <- "OP"
      } else if (length(empty) >= 3) {
        sched[idx, sw + empty[1:min(3, length(empty))]] <- "IP_SINGLE"
        if (length(empty) > 3) sched[idx, sw + empty[4:length(empty)]] <- "OP"
      }
      used[idx] <- TRUE
    }

    # Fill 1-week rotations
    single_rots <- list(
      list(rot = "rMICU", need = n_micu),
      list(rot = "Bronze", need = n_bronze),
      list(rot = "Cards", need = n_cards),
      list(rot = "Diamond", need = n_diamond),
      list(rot = "Gold", need = n_gold),
      list(rot = "Jeopardy", need = n_jeopardy)
    )

    for (wo in 1:6) {
      w <- sw + wo
      for (sr in single_rots) {
        avail <- which(sched[, w] == "IP_SINGLE")
        if (length(avail) > 0 && sr$need > 0) {
          scores <- annual[avail, sr$rot]
          avail <- avail[order(scores, runif(length(avail)))]
          n_assign <- min(sr$need, length(avail))
          for (k in seq_len(n_assign)) {
            sched[avail[k], w] <- sr$rot
            annual[avail[k], sr$rot] <- annual[avail[k], sr$rot] + 1
          }
        }
      }
    }
  }

  sched[sched == "IP_SINGLE"] <- "Elective"

  list(schedule = sched, ids = ids, pgy = pgy_vec, cpos = cpos,
       annual = annual, u_rots = u_rots, n_upper = n_upper)
}

# ═══════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════

compute_staffing <- function(result) {
  sched <- result$schedule
  u_rots <- result$u_rots
  TW <- ncol(sched)

  df <- data.frame(Week = 1:TW, Block = rep(1:8, each = 6), WIB = rep(1:6, 8))
  for (rn in names(u_rots)) {
    df[[rn]] <- sapply(1:TW, function(w) sum(sched[, w] == rn))
  }
  df$Total_IP <- rowSums(df[, names(u_rots), drop = FALSE])
  df$Total_Need <- sum(unlist(u_rots))
  df
}

compute_block_check <- function(result) {
  sched <- result$schedule
  ip_rots <- names(result$u_rots)
  n <- nrow(sched)

  df <- data.frame(Resident = result$ids, PGY = result$pgy, stringsAsFactors = FALSE)
  for (b in 1:8) {
    cols <- ((b - 1) * 6 + 1):(b * 6)
    df[[paste0("B", b)]] <- sapply(1:n, function(i) {
      cl <- sum(sched[i, cols] == "Clinic")
      ip <- sum(sched[i, cols] %in% ip_rots)
      op <- sum(sched[i, cols] %in% c("OP", "Elective"))
      if (cl == 1 && ip == 3 && op == 2) "OK" else paste0(cl, "C/", ip, "I/", op, "O")
    })
  }
  df
}

# ═══════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════

ui <- fluidPage(
  tags$head(tags$style(HTML("
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa; }
    .sidebar-panel { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
    .metric { padding: 18px; border-radius: 10px; text-align: center; color: white; margin: 5px; }
    .metric h3 { margin: 0; font-size: 28px; }
    .metric p { margin: 5px 0 0; font-size: 13px; opacity: 0.9; }
    .bg-green { background: #27ae60; }
    .bg-yellow { background: #f39c12; }
    .bg-red { background: #e74c3c; }
    .bg-blue { background: #2980b9; }
    h2 { color: #2c3e50; }
    .nav-tabs > li.active > a { border-top: 3px solid #2980b9; }
  "))),

  titlePanel(h2("Project Z \u2014 ABC Schedule Explorer")),

  sidebarLayout(
    sidebarPanel(
      width = 3,
      h4("Program Size"),
      fluidRow(
        column(4, numericInput("n_pgy3", "PGY3", 27, min = 1, max = 50)),
        column(4, numericInput("n_pgy2", "PGY2", 28, min = 1, max = 50)),
        column(4, numericInput("n_pgy1", "PGY1", 27, min = 1, max = 50))
      ),
      hr(),
      h4("Weekly IP Needs"),
      fluidRow(
        column(6, numericInput("n_sluh", "SLUH", 6, min = 1, max = 12)),
        column(6, numericInput("n_va", "VA", 5, min = 1, max = 10))
      ),
      fluidRow(
        column(6, numericInput("n_micu", "MICU", 4, min = 1, max = 8)),
        column(6, numericInput("n_nf", "NF", 5, min = 1, max = 8))
      ),
      fluidRow(
        column(6, numericInput("n_bronze", "Bronze", 2, min = 0, max = 4)),
        column(6, numericInput("n_cards", "Cards", 2, min = 0, max = 4))
      ),
      fluidRow(
        column(6, numericInput("n_diamond", "Diamond", 1, min = 0, max = 3)),
        column(6, numericInput("n_gold", "Gold", 1, min = 0, max = 3))
      ),
      fluidRow(
        column(6, numericInput("n_id", "ID", 1, min = 0, max = 3)),
        column(6, numericInput("n_jeopardy", "Jeop", 1, min = 0, max = 3))
      ),
      hr(),
      numericInput("seed", "Seed", 42, min = 1, max = 9999),
      actionButton("run", "Build Schedule",
                   class = "btn-primary", style = "width:100%; font-size:15px; padding:10px;")
    ),

    mainPanel(
      width = 9,
      fluidRow(
        column(3, uiOutput("m1")),
        column(3, uiOutput("m2")),
        column(3, uiOutput("m3")),
        column(3, uiOutput("m4"))
      ),
      br(),
      tabsetPanel(
        tabPanel("Staffing", br(),
                 p("Actual count per rotation per week. Red = gap, green = met."),
                 DTOutput("tbl_staff")),
        tabPanel("Schedule", br(),
                 selectInput("blk", "Block:", c("All" = 0, paste0("Block ", 1:8))),
                 DTOutput("tbl_sched")),
        tabPanel("Validation", br(),
                 p("Block constraint check: each cell should show 'OK' (1 Clinic + 3 IP + 2 OP)."),
                 DTOutput("tbl_block")),
        tabPanel("Balance", br(), DTOutput("tbl_annual")),
        tabPanel("Math", br(), uiOutput("math"))
      )
    )
  )
)

# ═══════════════════════════════════════════════════════════════════
# SERVER
# ═══════════════════════════════════════════════════════════════════

server <- function(input, output, session) {

  rv <- reactiveVal(build_schedule())

  observeEvent(input$run, {
    rv(build_schedule(
      n_pgy3 = input$n_pgy3, n_pgy2 = input$n_pgy2,
      n_sluh = input$n_sluh, n_va = input$n_va,
      n_micu = input$n_micu, n_nf = input$n_nf,
      n_bronze = input$n_bronze, n_cards = input$n_cards,
      n_diamond = input$n_diamond, n_gold = input$n_gold,
      n_id = input$n_id, n_jeopardy = input$n_jeopardy,
      seed = input$seed
    ))
  })

  sd <- reactive({ req(rv()); compute_staffing(rv()) })
  bd <- reactive({ req(rv()); compute_block_check(rv()) })

  output$m1 <- renderUI({
    n <- rv()$n_upper
    div(class = "metric bg-blue", h3(n), p("Upper-Level"))
  })
  output$m2 <- renderUI({
    ok <- sum(sd()$Total_IP >= sd()$Total_Need)
    cls <- if (ok >= 40) "bg-green" else if (ok >= 30) "bg-yellow" else "bg-red"
    div(class = paste("metric", cls), h3(paste0(ok, "/48")), p("Fully Staffed"))
  })
  output$m3 <- renderUI({
    viol <- sum(grepl("/", unlist(bd()[, -(1:2)])))
    cls <- if (viol == 0) "bg-green" else "bg-red"
    div(class = paste("metric", cls), h3(viol), p("Block Violations"))
  })
  output$m4 <- renderUI({
    gap <- sum(unlist(rv()$u_rots)) * 48 - rv()$n_upper * 24
    cls <- if (gap <= 0) "bg-green" else if (gap <= 30) "bg-yellow" else "bg-red"
    div(class = paste("metric", cls), h3(gap), p("IP-Week Deficit"))
  })

  output$tbl_staff <- renderDT({
    s <- sd()
    rots <- names(rv()$u_rots)
    needs <- unlist(rv()$u_rots)

    disp <- data.frame(Rotation = rots, Need = needs, stringsAsFactors = FALSE)
    for (w in 1:48) {
      nm <- paste0("B", s$Block[w], "W", s$WIB[w])
      disp[[nm]] <- sapply(rots, function(rn) s[[rn]][w])
    }

    datatable(disp, options = list(scrollX = TRUE, pageLength = 15, dom = "t",
              columnDefs = list(list(className = "dt-center", targets = "_all"))),
              rownames = FALSE)
  })

  output$tbl_sched <- renderDT({
    res <- rv()
    b <- as.integer(gsub("\\D", "", input$blk))
    if (is.na(b) || b == 0) cols <- 1:48 else cols <- ((b-1)*6+1):(b*6)

    disp <- data.frame(ID = res$ids, PGY = res$pgy, Pos = res$cpos, stringsAsFactors = FALSE)
    for (w in cols) {
      nm <- paste0("B", ((w-1)%/%6)+1, "W", ((w-1)%%6)+1)
      disp[[nm]] <- res$schedule[, w]
    }
    datatable(disp, options = list(scrollX = TRUE, pageLength = 20, dom = "ftip",
              columnDefs = list(list(className = "dt-center", targets = "_all"))),
              rownames = FALSE)
  })

  output$tbl_block <- renderDT({
    datatable(bd(), options = list(scrollX = TRUE, pageLength = 20, dom = "ftip",
              columnDefs = list(list(className = "dt-center", targets = "_all"))),
              rownames = FALSE)
  })

  output$tbl_annual <- renderDT({
    res <- rv()
    disp <- data.frame(ID = res$ids, PGY = res$pgy, as.data.frame(res$annual),
                       stringsAsFactors = FALSE)
    disp$Total_IP <- rowSums(res$annual)
    datatable(disp, options = list(scrollX = TRUE, pageLength = 20, dom = "ftip",
              columnDefs = list(list(className = "dt-center", targets = "_all"))),
              rownames = FALSE)
  })

  output$math <- renderUI({
    res <- rv()
    n <- res$n_upper
    avail <- n * 3 * 8
    need <- sum(unlist(res$u_rots)) * 48
    gap <- need - avail

    tagList(
      h3("Constraint Math"),
      h4("IP Supply & Demand"),
      p(paste0("Upper-level residents: ", n)),
      p(paste0("IP-weeks available: ", n, " x 3 IP/block x 8 blocks = ", avail)),
      p(paste0("IP-weeks needed: ", sum(unlist(res$u_rots)), " slots/wk x 48 wks = ", need)),
      p(strong(if (gap > 0) paste0("Deficit: ", gap, " weeks (~", round(gap/48,1), "/wk)")
               else "No deficit - surplus available")),

      h4("Per Resident Per Year"),
      p("IP: 24 weeks | OP: 16 weeks | Clinic: 8 weeks | Total: 48"),

      h4("3-Year ACGME Projection"),
      tags$ul(
        tags$li("Inpatient: 72 weeks (need >= 40)"),
        tags$li("Outpatient: 48 weeks (need >= 40)"),
        tags$li("ICU: ~10-12 weeks (need >= 12, achievable)"),
        tags$li("Elective: within OP allocation")
      ),

      h4("Rotation Durations"),
      tags$table(class = "table table-sm table-striped",
        tags$thead(tags$tr(tags$th("Rotation"), tags$th("Duration"), tags$th("Need/Wk"))),
        tags$tbody(
          tags$tr(tags$td("SLUH/VA/ID"), tags$td("3-wk continuous"), tags$td(paste(res$u_rots$rSLUH, "/", res$u_rots$rVA, "/", res$u_rots$ID))),
          tags$tr(tags$td("Night Float"), tags$td("2-wk continuous"), tags$td(res$u_rots$rNF)),
          tags$tr(tags$td("MICU/Bronze/Cards"), tags$td("1-wk intervals"), tags$td(paste(res$u_rots$rMICU, "/", res$u_rots$Bronze, "/", res$u_rots$Cards))),
          tags$tr(tags$td("Diamond/Gold/Jeopardy"), tags$td("1-wk"), tags$td(paste(res$u_rots$Diamond, "/", res$u_rots$Gold, "/", res$u_rots$Jeopardy)))
        )
      )
    )
  })
}

shinyApp(ui, server)
