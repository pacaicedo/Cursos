############################################################
# SHAP + LIME with a small psychology-style data set in R
#
# Data set:
#   datasets::attitude
#   An organizational psychology / employee attitude survey.
#
# The app predicts the overall department rating from survey
# dimensions, then explains one selected profile with:
#   - exact interventional SHAP values
#   - a LIME-style local weighted linear surrogate
#
# Run with:
#   install.packages(c("shiny", "plotly"))
#   shiny::runApp("psych_shap_lime_app.R")
############################################################

required_packages <- c("shiny", "plotly")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Please install the required package(s) first: install.packages(c(",
    paste(sprintf('"%s"', missing_packages), collapse = ", "),
    "))"
  )
}

library(shiny)
library(plotly)

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

fmt <- function(x, digits = 2) {
  formatC(x, format = "f", digits = digits)
}

signed_fmt <- function(x, digits = 2) {
  paste0(ifelse(x >= 0, "+", ""), fmt(x, digits))
}

clip_vec <- function(x, lower, upper) {
  pmin(upper, pmax(lower, x))
}

data("attitude", package = "datasets")
attitude_data <- datasets::attitude
attitude_data$department <- paste0("Dept ", seq_len(nrow(attitude_data)))

target_name <- "rating"
predictor_names <- c("complaints", "privileges", "learning", "raises", "critical", "advance")

feature_labels <- c(
  complaints = "Complaints handled well",
  privileges = "No special privileges",
  learning = "Learning opportunities",
  raises = "Raises for performance",
  critical = "Not overly critical",
  advance = "Advancement opportunities"
)

feature_help <- c(
  complaints = "How well employee complaints are handled.",
  privileges = "Whether special privileges are avoided.",
  learning = "How much opportunity employees have to learn.",
  raises = "Whether raises are based on performance.",
  critical = "Whether supervision is not too critical.",
  advance = "How much opportunity there is for advancement."
)

feature_min <- vapply(attitude_data[, predictor_names], min, numeric(1))
feature_max <- vapply(attitude_data[, predictor_names], max, numeric(1))
feature_mean <- vapply(attitude_data[, predictor_names], mean, numeric(1))
feature_sd <- vapply(attitude_data[, predictor_names], sd, numeric(1))

# A small nonlinear predictive model.
# We treat this as a black box in the app: the explanation methods only ask it
# for predictions. This keeps the demo dependency-light while still showing
# nonlinearity, interactions, and local approximation error.
black_box_formula <- rating ~
  complaints + privileges + learning + raises + critical + advance +
  I(complaints^2) + I(learning^2) + I(raises^2) +
  complaints:learning + complaints:raises + privileges:advance

black_box_fit <- lm(black_box_formula, data = attitude_data)

black_box_predict <- function(new_data) {
  new_df <- as.data.frame(new_data[, predictor_names, drop = FALSE])
  as.numeric(predict(black_box_fit, newdata = new_df))
}

training_predictions <- black_box_predict(attitude_data)
baseline_prediction <- mean(training_predictions)

department_choices <- setNames(
  seq_len(nrow(attitude_data)),
  paste0(
    attitude_data$department,
    " - observed rating ",
    attitude_data$rating
  )
)

default_department <- 10

make_case_df <- function(values_named) {
  case_df <- as.data.frame(as.list(values_named[predictor_names]))
  names(case_df) <- predictor_names
  case_df
}

case_hover_text <- function(case_df, prefix = "Current profile") {
  paste0(
    "<b>", prefix, "</b><br>",
    paste(
      paste0(feature_labels[predictor_names], ": ", fmt(as.numeric(case_df[1, predictor_names]), 1)),
      collapse = "<br>"
    )
  )
}

row_hover_text <- vapply(
  seq_len(nrow(attitude_data)),
  function(row_idx) {
    paste0(
      "<b>", attitude_data$department[[row_idx]], "</b><br>",
      "Observed rating: ", attitude_data$rating[[row_idx]], "<br>",
      "Black-box prediction: ", fmt(training_predictions[[row_idx]]), "<br>",
      paste(
        vapply(
          predictor_names,
          function(nm) paste0(feature_labels[[nm]], ": ", attitude_data[[nm]][[row_idx]]),
          character(1)
        ),
        collapse = "<br>"
      )
    )
  },
  character(1)
)

coalition_value <- function(mask, case_df, background_df) {
  p <- length(predictor_names)
  active_features <- which(as.integer(intToBits(mask))[seq_len(p)] == 1L)
  query_df <- background_df[, predictor_names, drop = FALSE]

  if (length(active_features) > 0) {
    for (feature_idx in active_features) {
      feature_name <- predictor_names[[feature_idx]]
      query_df[[feature_name]] <- case_df[[feature_name]]
    }
  }

  mean(black_box_predict(query_df))
}

calculate_shap_values <- function(case_df, background_df = attitude_data) {
  p <- length(predictor_names)
  n_masks <- 2^p
  coalition_values <- numeric(n_masks)

  for (mask in 0:(n_masks - 1)) {
    coalition_values[[mask + 1]] <- coalition_value(mask, case_df, background_df)
  }

  shap_values <- numeric(p)
  names(shap_values) <- predictor_names

  for (feature_idx in seq_len(p)) {
    feature_bit <- bitwShiftL(1L, feature_idx - 1L)

    for (mask in 0:(n_masks - 1)) {
      has_feature <- bitwAnd(mask, feature_bit) != 0L

      if (!has_feature) {
        subset_size <- sum(as.integer(intToBits(mask))[seq_len(p)])
        weight <- factorial(subset_size) * factorial(p - subset_size - 1) / factorial(p)
        mask_with_feature <- bitwOr(mask, feature_bit)
        marginal_gain <- coalition_values[[mask_with_feature + 1]] - coalition_values[[mask + 1]]
        shap_values[[feature_idx]] <- shap_values[[feature_idx]] + weight * marginal_gain
      }
    }
  }

  case_prediction <- black_box_predict(case_df)
  shap_df <- data.frame(
    feature = predictor_names,
    label = unname(feature_labels[predictor_names]),
    value = as.numeric(case_df[1, predictor_names]),
    shap = as.numeric(shap_values[predictor_names]),
    abs_shap = abs(as.numeric(shap_values[predictor_names])),
    direction = ifelse(shap_values[predictor_names] >= 0, "pushes up", "pushes down"),
    stringsAsFactors = FALSE
  )

  list(
    baseline = coalition_values[[1]],
    prediction = case_prediction,
    reconstruction = coalition_values[[1]] + sum(shap_values),
    values = shap_df,
    coalition_values = coalition_values
  )
}

scale_predictors <- function(predictor_df) {
  scaled_matrix <- sweep(as.matrix(predictor_df[, predictor_names, drop = FALSE]), 2, feature_mean, "-")
  scaled_matrix <- sweep(scaled_matrix, 2, feature_sd, "/")
  scaled_df <- as.data.frame(scaled_matrix)
  names(scaled_df) <- predictor_names
  scaled_df
}

calculate_lime_values <- function(case_df, n_samples, kernel_width, perturb_scale, seed_value) {
  set.seed(seed_value)

  p <- length(predictor_names)
  case_values <- as.numeric(case_df[1, predictor_names])
  names(case_values) <- predictor_names

  sample_matrix <- matrix(NA_real_, nrow = n_samples, ncol = p)
  colnames(sample_matrix) <- predictor_names

  for (feature_idx in seq_len(p)) {
    feature_name <- predictor_names[[feature_idx]]
    raw_values <- rnorm(
      n_samples,
      mean = case_values[[feature_name]],
      sd = perturb_scale * feature_sd[[feature_name]]
    )
    sample_matrix[, feature_idx] <- clip_vec(
      raw_values,
      lower = feature_min[[feature_name]],
      upper = feature_max[[feature_name]]
    )
  }

  # Make sure the exact case is present in the neighborhood.
  sample_matrix[1, ] <- case_values[predictor_names]

  sample_df <- as.data.frame(sample_matrix)
  names(sample_df) <- predictor_names

  scaled_samples <- scale_predictors(sample_df)
  case_scaled <- (case_values[predictor_names] - feature_mean[predictor_names]) / feature_sd[predictor_names]

  distance_vec <- sqrt(rowSums(sweep(as.matrix(scaled_samples), 2, case_scaled, "-")^2))
  weight_vec <- exp(-(distance_vec^2) / (kernel_width^2))
  weight_vec[1] <- 1

  black_box_vec <- black_box_predict(sample_df)
  fit_df <- cbind(prediction = black_box_vec, scaled_samples)
  local_fit <- lm(prediction ~ ., data = fit_df, weights = weight_vec)

  coef_vec <- coef(local_fit)
  intercept <- unname(coef_vec[[1]])
  beta_vec <- coef_vec[predictor_names]
  beta_vec[is.na(beta_vec)] <- 0

  scaled_matrix <- as.matrix(scaled_samples[, predictor_names, drop = FALSE])
  center_scaled <- colSums(scaled_matrix * weight_vec) / sum(weight_vec)
  center_original <- center_scaled * feature_sd[predictor_names] + feature_mean[predictor_names]

  contributions <- beta_vec * (case_scaled[predictor_names] - center_scaled[predictor_names])
  local_baseline <- intercept + sum(beta_vec * center_scaled[predictor_names])
  surrogate_prediction <- intercept + sum(beta_vec * case_scaled[predictor_names])

  lime_pred_samples <- as.numeric(predict(local_fit, newdata = fit_df))
  weighted_mean <- weighted.mean(black_box_vec, weight_vec)
  weighted_rss <- sum(weight_vec * (black_box_vec - lime_pred_samples)^2)
  weighted_tss <- sum(weight_vec * (black_box_vec - weighted_mean)^2)
  weighted_r2 <- ifelse(weighted_tss > 0, 1 - weighted_rss / weighted_tss, NA_real_)

  contribution_df <- data.frame(
    feature = predictor_names,
    label = unname(feature_labels[predictor_names]),
    selected_value = as.numeric(case_values[predictor_names]),
    local_center = as.numeric(center_original[predictor_names]),
    local_slope = as.numeric(beta_vec[predictor_names]),
    contribution = as.numeric(contributions[predictor_names]),
    abs_contribution = abs(as.numeric(contributions[predictor_names])),
    direction = ifelse(contributions[predictor_names] >= 0, "pushes up", "pushes down"),
    stringsAsFactors = FALSE
  )

  sample_plot_df <- cbind(
    sample_df,
    data.frame(
      distance = distance_vec,
      weight = weight_vec,
      black_box_prediction = black_box_vec,
      lime_prediction = lime_pred_samples,
      row_type = ifelse(seq_len(n_samples) == 1, "selected case", "perturbed case"),
      stringsAsFactors = FALSE
    )
  )

  list(
    samples = sample_plot_df,
    contributions = contribution_df,
    local_baseline = local_baseline,
    surrogate_prediction = surrogate_prediction,
    black_box_prediction = black_box_predict(case_df),
    weighted_r2 = weighted_r2,
    coefficients = coef_vec
  )
}

plotly_empty_message <- function(message_text) {
  plotly::plot_ly() |>
    plotly::layout(
      xaxis = list(visible = FALSE),
      yaxis = list(visible = FALSE),
      annotations = list(
        list(
          text = message_text,
          x = 0.5,
          y = 0.5,
          showarrow = FALSE,
          font = list(size = 16)
        )
      )
    )
}

app_css <- "
body {
  background: #f8fafc;
  color: #0f172a;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.hero {
  background:
    radial-gradient(circle at 12% 20%, rgba(56, 189, 248, 0.28), transparent 28%),
    radial-gradient(circle at 88% 15%, rgba(250, 204, 21, 0.25), transparent 25%),
    linear-gradient(135deg, #0f172a 0%, #1d4ed8 58%, #0f766e 100%);
  color: white;
  border-radius: 24px;
  padding: 28px 34px;
  margin: 18px 0 22px 0;
  box-shadow: 0 20px 45px rgba(15, 23, 42, 0.20);
}
.hero h1 {
  margin: 0 0 8px 0;
  font-size: 34px;
  font-weight: 850;
  letter-spacing: -0.035em;
}
.hero p {
  font-size: 17px;
  line-height: 1.55;
  max-width: 1100px;
  margin: 0;
  color: #dbeafe;
}
.well {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 18px;
  box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.metric-card {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 18px;
  padding: 15px 16px;
  min-height: 118px;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
}
.metric-label {
  color: #64748b;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 800;
}
.metric-value {
  font-size: 30px;
  font-weight: 850;
  letter-spacing: -0.04em;
  margin-top: 6px;
  color: #0f172a;
}
.metric-note {
  color: #475569;
  font-size: 13px;
  margin-top: 4px;
  line-height: 1.35;
}
.explain-card {
  background: #ffffff;
  border-left: 6px solid #2563eb;
  border-radius: 14px;
  padding: 16px 18px;
  margin: 12px 0 18px 0;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
  line-height: 1.55;
}
.explain-card h3 {
  margin-top: 0;
}
.soft-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 12px;
  line-height: 1.50;
}
.mini-note {
  color: #475569;
  font-size: 13px;
  line-height: 1.45;
}
.control-label {
  font-weight: 700;
  color: #1e293b;
}
.nav-tabs > li > a {
  border-radius: 12px 12px 0 0;
  font-weight: 750;
}
.tab-content {
  background: white;
  border: 1px solid #ddd;
  border-top: 0;
  border-radius: 0 0 18px 18px;
  padding: 18px;
  margin-bottom: 30px;
}
table {
  font-size: 14px;
}
.formula-box {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
  background: #0f172a;
  color: #dbeafe;
  padding: 14px;
  border-radius: 14px;
  overflow-x: auto;
  white-space: pre-wrap;
}
@media (max-width: 1100px) {
  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
@media (max-width: 700px) {
  .metric-grid {
    grid-template-columns: 1fr;
  }
}
"

ui <- fluidPage(
  tags$head(tags$style(HTML(app_css))),
  div(
    class = "hero",
    h1("SHAP and LIME, explained with an R psychology data set"),
    p(
      "This app uses the built-in R data set ",
      tags$code("datasets::attitude"),
      ", an organizational psychology survey of employee attitudes across 30 departments. ",
      "We predict the overall attitude rating, then explain one profile with SHAP and LIME. ",
      "Every plot is interactive: hover over points and bars to see the values."
    )
  ),
  sidebarLayout(
    sidebarPanel(
      width = 3,
      h3("1. Choose a profile"),
      selectInput(
        "selected_row",
        "Start from a department",
        choices = department_choices,
        selected = default_department
      ),
      div(class = "mini-note", "Changing the department resets the sliders. Moving a slider creates a custom profile."),
      uiOutput("feature_sliders"),
      hr(),
      h3("2. Pick display features"),
      selectInput(
        "plot_x",
        "X-axis feature",
        choices = setNames(predictor_names, unname(feature_labels[predictor_names])),
        selected = "complaints"
      ),
      selectInput(
        "plot_y",
        "Y-axis feature",
        choices = setNames(predictor_names, unname(feature_labels[predictor_names])),
        selected = "learning"
      ),
      hr(),
      h3("3. Tune LIME"),
      sliderInput("lime_n", "Perturbed profiles", min = 200, max = 3000, value = 900, step = 100),
      sliderInput("lime_kernel", "Local neighborhood width", min = 0.20, max = 4.00, value = 1.25, step = 0.05),
      sliderInput("lime_perturb", "Perturbation spread", min = 0.20, max = 2.00, value = 0.85, step = 0.05),
      numericInput("lime_seed", "Random seed", value = 123, min = 1, max = 999999, step = 1),
      checkboxInput("show_formula", "Show the teaching model formula", value = FALSE),
      hr(),
      div(
        class = "mini-note",
        strong("Tip: "),
        "Make the LIME neighborhood very small to see a highly local explanation. Make it wide to see the surrogate average over more of the model surface."
      )
    ),
    mainPanel(
      width = 9,
      uiOutput("metric_cards"),
      tabsetPanel(
        tabPanel(
          "Data and model",
          div(class = "explain-card", uiOutput("data_model_text")),
          conditionalPanel(
            condition = "input.show_formula == true",
            div(class = "formula-box", textOutput("formula_text"))
          ),
          fluidRow(
            column(width = 6, plotlyOutput("data_scatter", height = "430px")),
            column(width = 6, plotlyOutput("model_surface", height = "430px"))
          ),
          div(class = "soft-card", tableOutput("current_profile_table"))
        ),
        tabPanel(
          "SHAP",
          div(class = "explain-card", uiOutput("shap_text")),
          plotlyOutput("shap_waterfall", height = "430px"),
          fluidRow(
            column(width = 7, plotlyOutput("shap_bar", height = "390px")),
            column(width = 5, div(class = "soft-card", tableOutput("shap_table")))
          )
        ),
        tabPanel(
          "LIME",
          div(class = "explain-card", uiOutput("lime_text")),
          fluidRow(
            column(width = 6, plotlyOutput("lime_neighborhood", height = "430px")),
            column(width = 6, plotlyOutput("lime_waterfall", height = "430px"))
          ),
          fluidRow(
            column(width = 6, plotlyOutput("lime_fit", height = "390px")),
            column(width = 6, div(class = "soft-card", tableOutput("lime_table")))
          )
        ),
        tabPanel(
          "Compare",
          div(class = "explain-card", uiOutput("compare_text")),
          fluidRow(
            column(width = 6, plotlyOutput("compare_contributions", height = "430px")),
            column(width = 6, plotlyOutput("compare_predictions", height = "430px"))
          )
        ),
        tabPanel(
          "Plain-English guide",
          div(
            class = "explain-card",
            h3("The simplest way to think about SHAP"),
            p(strong("SHAP starts with the average prediction."), "Then it asks how much each feature fairly adds or subtracts as we move from an average department to this selected profile."),
            p("The key idea is fairness: a feature gets credit based on its average marginal contribution across all possible orders in which features could enter the explanation."),
            h3("The simplest way to think about LIME"),
            p(strong("LIME builds a simple local explanation."), "It creates many nearby fake profiles, asks the black-box model to score them, weights the closest profiles most heavily, and fits a small linear model in that neighborhood."),
            p("The local linear model becomes the explanation. If the black-box model is very curved nearby, LIME may only be approximate."),
            h3("Main difference"),
            p(strong("SHAP is an additive attribution method."), "Its pieces are designed to add up to the black-box prediction."),
            p(strong("LIME is a local approximation method."), "Its explanation depends on the neighborhood width, perturbations, and random seed.")
          )
        )
      )
    )
  )
)

server <- function(input, output, session) {
  output$feature_sliders <- renderUI({
    selected_idx <- as.integer(input$selected_row %||% default_department)

    tagList(
      lapply(
        predictor_names,
        function(feature_name) {
          sliderInput(
            inputId = paste0("val_", feature_name),
            label = paste0(feature_labels[[feature_name]], " (", feature_name, ")"),
            min = floor(feature_min[[feature_name]]),
            max = ceiling(feature_max[[feature_name]]),
            value = attitude_data[[feature_name]][[selected_idx]],
            step = 1
          )
        }
      )
    )
  })

  observeEvent(input$selected_row,
    {
      selected_idx <- as.integer(input$selected_row)

      for (feature_name in predictor_names) {
        updateSliderInput(
          session,
          inputId = paste0("val_", feature_name),
          value = attitude_data[[feature_name]][[selected_idx]]
        )
      }
    },
    ignoreInit = TRUE
  )

  selected_case <- reactive({
    values_named <- vapply(
      predictor_names,
      function(feature_name) {
        input[[paste0("val_", feature_name)]] %||% attitude_data[[feature_name]][[default_department]]
      },
      numeric(1)
    )
    names(values_named) <- predictor_names
    make_case_df(values_named)
  })

  selected_prediction <- reactive({
    black_box_predict(selected_case())
  })

  shap_result <- reactive({
    calculate_shap_values(selected_case(), attitude_data)
  })

  lime_result <- reactive({
    calculate_lime_values(
      case_df = selected_case(),
      n_samples = input$lime_n,
      kernel_width = input$lime_kernel,
      perturb_scale = input$lime_perturb,
      seed_value = input$lime_seed
    )
  })

  top_shap_feature <- reactive({
    shap_df <- shap_result()$values
    shap_df[which.max(shap_df$abs_shap), , drop = FALSE]
  })

  top_lime_feature <- reactive({
    lime_df <- lime_result()$contributions
    lime_df[which.max(lime_df$abs_contribution), , drop = FALSE]
  })

  output$metric_cards <- renderUI({
    shap_res <- shap_result()
    lime_res <- lime_result()
    lime_error <- lime_res$surrogate_prediction - shap_res$prediction

    div(
      class = "metric-grid",
      div(
        class = "metric-card",
        div(class = "metric-label", "Black-box prediction"),
        div(class = "metric-value", fmt(shap_res$prediction)),
        div(class = "metric-note", "Predicted overall attitude rating for the current profile.")
      ),
      div(
        class = "metric-card",
        div(class = "metric-label", "Average baseline"),
        div(class = "metric-value", fmt(shap_res$baseline)),
        div(class = "metric-note", "The average prediction across all 30 departments.")
      ),
      div(
        class = "metric-card",
        div(class = "metric-label", "SHAP sum"),
        div(class = "metric-value", fmt(shap_res$reconstruction)),
        div(class = "metric-note", "Baseline plus SHAP values. This equals the model prediction.")
      ),
      div(
        class = "metric-card",
        div(class = "metric-label", "LIME surrogate"),
        div(class = "metric-value", fmt(lime_res$surrogate_prediction)),
        div(class = "metric-note", paste0("Surrogate error: ", signed_fmt(lime_error), ". Local R-squared: ", fmt(lime_res$weighted_r2, 2), "."))
      )
    )
  })

  output$formula_text <- renderText({
    paste(deparse(black_box_formula), collapse = "\n")
  })

  output$data_model_text <- renderUI({
    HTML(paste0(
      "<b>Data:</b> <code>datasets::attitude</code> has 30 departments and survey scores from an employee-attitude study. ",
      "The outcome is <b>rating</b>, the overall department rating. ",
      "<br><br><b>Teaching model:</b> we fit a small nonlinear regression and then treat it as a black box. ",
      "The plots below show the observed data and a two-feature slice of the prediction surface. ",
      "Hover over points to inspect departments and predictions."
    ))
  })

  output$shap_text <- renderUI({
    shap_res <- shap_result()
    top_feature <- top_shap_feature()

    HTML(paste0(
      "<b>SHAP question:</b> starting from the average prediction of <b>",
      fmt(shap_res$baseline), "</b>, how much does each feature fairly move us to this profile's prediction of <b>",
      fmt(shap_res$prediction), "</b>? ",
      "For this profile, the largest SHAP contributor is <b>",
      top_feature$label, "</b>, which <b>", top_feature$direction, "</b> the prediction by <b>",
      signed_fmt(top_feature$shap), "</b>. ",
      "Hover over the waterfall or bars for exact values."
    ))
  })

  output$lime_text <- renderUI({
    lime_res <- lime_result()
    top_feature <- top_lime_feature()
    lime_error <- lime_res$surrogate_prediction - lime_res$black_box_prediction

    HTML(paste0(
      "<b>LIME question:</b> what simple linear model behaves like the black box near this one profile? ",
      "The app generates <b>", input$lime_n, "</b> nearby profiles, weights closer ones more heavily, ",
      "and fits a weighted linear surrogate. ",
      "The largest LIME contribution is <b>", top_feature$label, "</b>, which <b>",
      top_feature$direction, "</b> the surrogate by <b>", signed_fmt(top_feature$contribution), "</b>. ",
      "The surrogate is currently <b>", signed_fmt(lime_error), "</b> away from the black-box prediction."
    ))
  })

  output$compare_text <- renderUI({
    shap_top <- top_shap_feature()
    lime_top <- top_lime_feature()

    HTML(paste0(
      "<b>SHAP and LIME can disagree because they answer different questions.</b> ",
      "SHAP fairly divides the movement from the average prediction to the selected prediction. ",
      "LIME explains a local straight-line approximation around the selected profile. ",
      "Here, SHAP's largest feature is <b>", shap_top$label,
      "</b>, while LIME's largest feature is <b>", lime_top$label, "</b>. ",
      "Try changing the LIME neighborhood width and watch the LIME explanation move."
    ))
  })

  output$current_profile_table <- renderTable(
    {
      case_df <- selected_case()
      data.frame(
        Feature = unname(feature_labels[predictor_names]),
        Variable = predictor_names,
        `Current value` = as.numeric(case_df[1, predictor_names]),
        `Data min` = as.numeric(feature_min[predictor_names]),
        `Data max` = as.numeric(feature_max[predictor_names]),
        Meaning = unname(feature_help[predictor_names]),
        check.names = FALSE
      )
    },
    striped = TRUE,
    bordered = TRUE,
    spacing = "s"
  )

  output$data_scatter <- renderPlotly({
    x_feature <- input$plot_x
    y_feature <- input$plot_y

    if (identical(x_feature, y_feature)) {
      return(plotly_empty_message("Choose two different features for the X and Y axes."))
    }

    case_df <- selected_case()

    plotly::plot_ly(
      type = "scatter",
      mode = "markers",
      x = attitude_data[[x_feature]],
      y = attitude_data[[y_feature]],
      text = row_hover_text,
      hoverinfo = "text",
      marker = list(
        size = 12,
        color = attitude_data$rating,
        colorscale = "Viridis",
        showscale = TRUE,
        colorbar = list(title = "Observed<br>rating"),
        line = list(color = "#0f172a", width = 1)
      ),
      name = "Observed departments"
    ) |>
      plotly::add_markers(
        x = case_df[[x_feature]],
        y = case_df[[y_feature]],
        text = paste0(case_hover_text(case_df), "<br>Predicted rating: ", fmt(selected_prediction())),
        hoverinfo = "text",
        marker = list(
          size = 21,
          color = "#facc15",
          symbol = "star",
          line = list(color = "#111827", width = 2)
        ),
        name = "Current profile",
        inherit = FALSE
      ) |>
      plotly::layout(
        title = "Observed departments",
        xaxis = list(title = feature_labels[[x_feature]]),
        yaxis = list(title = feature_labels[[y_feature]]),
        legend = list(orientation = "h", x = 0, y = -0.18),
        margin = list(l = 70, r = 30, b = 80, t = 55)
      )
  })

  output$model_surface <- renderPlotly({
    x_feature <- input$plot_x
    y_feature <- input$plot_y

    if (identical(x_feature, y_feature)) {
      return(plotly_empty_message("Choose two different features for the X and Y axes."))
    }

    case_df <- selected_case()
    x_seq <- seq(feature_min[[x_feature]], feature_max[[x_feature]], length.out = 55)
    y_seq <- seq(feature_min[[y_feature]], feature_max[[y_feature]], length.out = 55)
    grid_df <- expand.grid(x_value = x_seq, y_value = y_seq)

    predict_grid <- case_df[rep(1, nrow(grid_df)), predictor_names, drop = FALSE]
    predict_grid[[x_feature]] <- grid_df$x_value
    predict_grid[[y_feature]] <- grid_df$y_value
    grid_prediction <- black_box_predict(predict_grid)
    z_matrix <- matrix(grid_prediction, nrow = length(x_seq), ncol = length(y_seq))

    surface_hover <- paste0(
      feature_labels[[x_feature]], ": %{x:.1f}<br>",
      feature_labels[[y_feature]], ": %{y:.1f}<br>",
      "Black-box prediction: %{z:.2f}<extra></extra>"
    )

    plotly::plot_ly() |>
      plotly::add_heatmap(
        x = x_seq,
        y = y_seq,
        z = t(z_matrix),
        colors = c("#eff6ff", "#93c5fd", "#2563eb", "#1e3a8a", "#0f172a"),
        hovertemplate = surface_hover,
        colorbar = list(title = "Predicted<br>rating"),
        name = "Prediction surface"
      ) |>
      plotly::add_markers(
        x = attitude_data[[x_feature]],
        y = attitude_data[[y_feature]],
        text = row_hover_text,
        hoverinfo = "text",
        marker = list(
          size = 8,
          color = "white",
          opacity = 0.72,
          line = list(color = "#0f172a", width = 1)
        ),
        name = "Observed departments",
        inherit = FALSE
      ) |>
      plotly::add_markers(
        x = case_df[[x_feature]],
        y = case_df[[y_feature]],
        text = paste0(case_hover_text(case_df), "<br>Predicted rating: ", fmt(selected_prediction())),
        hoverinfo = "text",
        marker = list(
          size = 22,
          color = "#facc15",
          symbol = "star",
          line = list(color = "#111827", width = 2)
        ),
        name = "Current profile",
        inherit = FALSE
      ) |>
      plotly::layout(
        title = "Two-feature slice of the black-box model",
        xaxis = list(title = feature_labels[[x_feature]]),
        yaxis = list(title = feature_labels[[y_feature]]),
        legend = list(orientation = "h", x = 0, y = -0.18),
        margin = list(l = 70, r = 30, b = 80, t = 55)
      )
  })

  output$shap_waterfall <- renderPlotly({
    shap_res <- shap_result()
    shap_df <- shap_res$values[order(shap_res$values$abs_shap, decreasing = TRUE), ]

    x_labels <- c("Average baseline", shap_df$label, "Prediction")
    y_values <- c(shap_res$baseline, shap_df$shap, 0)
    measures <- c("absolute", rep("relative", nrow(shap_df)), "total")
    hover_text <- c(
      paste0("<b>Average baseline</b><br>Mean prediction: ", fmt(shap_res$baseline)),
      paste0(
        "<b>", shap_df$label, "</b><br>",
        "Current value: ", fmt(shap_df$value, 1), "<br>",
        "SHAP value: ", signed_fmt(shap_df$shap), "<br>",
        "Direction: ", shap_df$direction
      ),
      paste0("<b>Final prediction</b><br>Black-box prediction: ", fmt(shap_res$prediction))
    )

    plotly::plot_ly(
      type = "waterfall",
      x = x_labels,
      y = y_values,
      measure = measures,
      text = c(fmt(shap_res$baseline), signed_fmt(shap_df$shap), fmt(shap_res$prediction)),
      textposition = "outside",
      hovertext = hover_text,
      hoverinfo = "text",
      increasing = list(marker = list(color = "#16a34a")),
      decreasing = list(marker = list(color = "#dc2626")),
      totals = list(marker = list(color = "#0f172a")),
      connector = list(line = list(color = "#94a3b8"))
    ) |>
      plotly::layout(
        title = "SHAP waterfall: average prediction to selected prediction",
        yaxis = list(title = "Predicted rating"),
        xaxis = list(title = ""),
        margin = list(l = 70, r = 30, b = 100, t = 55)
      )
  })

  output$shap_bar <- renderPlotly({
    shap_df <- shap_result()$values
    shap_df <- shap_df[order(shap_df$shap), ]
    shap_df$label <- factor(shap_df$label, levels = shap_df$label)
    bar_colors <- ifelse(shap_df$shap >= 0, "#16a34a", "#dc2626")

    hover_text <- paste0(
      "<b>", shap_df$label, "</b><br>",
      "Current value: ", fmt(shap_df$value, 1), "<br>",
      "SHAP value: ", signed_fmt(shap_df$shap), "<br>",
      "Meaning: ", shap_df$direction, " relative to the average profile"
    )

    plotly::plot_ly(
      data = shap_df,
      type = "bar",
      orientation = "h",
      x = ~shap,
      y = ~label,
      text = hover_text,
      hoverinfo = "text",
      marker = list(color = bar_colors)
    ) |>
      plotly::layout(
        title = "SHAP values by feature",
        xaxis = list(title = "Contribution to prediction", zeroline = TRUE),
        yaxis = list(title = ""),
        showlegend = FALSE,
        margin = list(l = 150, r = 30, b = 60, t = 55)
      )
  })

  output$shap_table <- renderTable(
    {
      shap_df <- shap_result()$values
      shap_df <- shap_df[order(shap_df$abs_shap, decreasing = TRUE), ]
      data.frame(
        Feature = shap_df$label,
        Value = fmt(shap_df$value, 1),
        SHAP = signed_fmt(shap_df$shap),
        Direction = shap_df$direction,
        check.names = FALSE
      )
    },
    striped = TRUE,
    bordered = TRUE,
    spacing = "s"
  )

  output$lime_neighborhood <- renderPlotly({
    x_feature <- input$plot_x
    y_feature <- input$plot_y

    if (identical(x_feature, y_feature)) {
      return(plotly_empty_message("Choose two different features for the X and Y axes."))
    }

    lime_res <- lime_result()
    sample_df <- lime_res$samples
    case_df <- selected_case()
    marker_size <- 6 + 16 * sample_df$weight / max(sample_df$weight)

    hover_text <- paste0(
      "<b>", sample_df$row_type, "</b><br>",
      feature_labels[[x_feature]], ": ", fmt(sample_df[[x_feature]], 1), "<br>",
      feature_labels[[y_feature]], ": ", fmt(sample_df[[y_feature]], 1), "<br>",
      "Distance from profile: ", fmt(sample_df$distance, 2), "<br>",
      "LIME weight: ", fmt(sample_df$weight, 3), "<br>",
      "Black-box prediction: ", fmt(sample_df$black_box_prediction), "<br>",
      "LIME surrogate prediction: ", fmt(sample_df$lime_prediction)
    )

    plotly::plot_ly(
      type = "scatter",
      mode = "markers",
      x = sample_df[[x_feature]],
      y = sample_df[[y_feature]],
      text = hover_text,
      hoverinfo = "text",
      marker = list(
        size = marker_size,
        color = sample_df$weight,
        colorscale = "Viridis",
        showscale = TRUE,
        colorbar = list(title = "LIME<br>weight"),
        opacity = 0.58,
        line = list(color = "rgba(15, 23, 42, 0.25)", width = 0.5)
      ),
      name = "Perturbed profiles"
    ) |>
      plotly::add_markers(
        x = case_df[[x_feature]],
        y = case_df[[y_feature]],
        text = paste0(case_hover_text(case_df), "<br>Black-box prediction: ", fmt(selected_prediction())),
        hoverinfo = "text",
        marker = list(
          size = 22,
          color = "#facc15",
          symbol = "star",
          line = list(color = "#111827", width = 2)
        ),
        name = "Current profile",
        inherit = FALSE
      ) |>
      plotly::layout(
        title = "LIME neighborhood: closer points get more weight",
        xaxis = list(title = feature_labels[[x_feature]]),
        yaxis = list(title = feature_labels[[y_feature]]),
        legend = list(orientation = "h", x = 0, y = -0.18),
        margin = list(l = 70, r = 30, b = 80, t = 55)
      )
  })

  output$lime_waterfall <- renderPlotly({
    lime_res <- lime_result()
    lime_df <- lime_res$contributions[order(lime_res$contributions$abs_contribution, decreasing = TRUE), ]

    x_labels <- c("Local baseline", lime_df$label, "LIME prediction")
    y_values <- c(lime_res$local_baseline, lime_df$contribution, 0)
    measures <- c("absolute", rep("relative", nrow(lime_df)), "total")

    hover_text <- c(
      paste0("<b>Local baseline</b><br>Surrogate prediction at weighted local center: ", fmt(lime_res$local_baseline)),
      paste0(
        "<b>", lime_df$label, "</b><br>",
        "Selected value: ", fmt(lime_df$selected_value, 1), "<br>",
        "Weighted local center: ", fmt(lime_df$local_center, 1), "<br>",
        "Local contribution: ", signed_fmt(lime_df$contribution), "<br>",
        "Local slope: ", signed_fmt(lime_df$local_slope)
      ),
      paste0("<b>LIME surrogate prediction</b><br>", fmt(lime_res$surrogate_prediction))
    )

    plotly::plot_ly(
      type = "waterfall",
      x = x_labels,
      y = y_values,
      measure = measures,
      text = c(fmt(lime_res$local_baseline), signed_fmt(lime_df$contribution), fmt(lime_res$surrogate_prediction)),
      textposition = "outside",
      hovertext = hover_text,
      hoverinfo = "text",
      increasing = list(marker = list(color = "#16a34a")),
      decreasing = list(marker = list(color = "#dc2626")),
      totals = list(marker = list(color = "#f97316")),
      connector = list(line = list(color = "#94a3b8"))
    ) |>
      plotly::layout(
        title = "LIME waterfall: local surrogate pieces",
        yaxis = list(title = "Predicted rating"),
        xaxis = list(title = ""),
        margin = list(l = 70, r = 30, b = 100, t = 55)
      )
  })

  output$lime_fit <- renderPlotly({
    lime_res <- lime_result()
    sample_df <- lime_res$samples
    axis_range <- range(c(sample_df$black_box_prediction, sample_df$lime_prediction), finite = TRUE)
    pad <- max(1, diff(axis_range) * 0.08)
    axis_range <- c(axis_range[1] - pad, axis_range[2] + pad)

    hover_text <- paste0(
      "<b>", sample_df$row_type, "</b><br>",
      "Black-box prediction: ", fmt(sample_df$black_box_prediction), "<br>",
      "LIME prediction: ", fmt(sample_df$lime_prediction), "<br>",
      "Error: ", signed_fmt(sample_df$lime_prediction - sample_df$black_box_prediction), "<br>",
      "Weight: ", fmt(sample_df$weight, 3)
    )

    plotly::plot_ly(
      type = "scatter",
      mode = "markers",
      x = sample_df$black_box_prediction,
      y = sample_df$lime_prediction,
      text = hover_text,
      hoverinfo = "text",
      marker = list(
        size = 7 + 14 * sample_df$weight / max(sample_df$weight),
        color = sample_df$weight,
        colorscale = "Viridis",
        showscale = TRUE,
        colorbar = list(title = "LIME<br>weight"),
        opacity = 0.62
      ),
      name = "Perturbed profiles"
    ) |>
      plotly::add_lines(
        x = axis_range,
        y = axis_range,
        line = list(color = "#dc2626", dash = "dash", width = 2),
        name = "Perfect mimic",
        inherit = FALSE
      ) |>
      plotly::layout(
        title = paste0("Local fidelity: weighted R-squared = ", fmt(lime_res$weighted_r2, 2)),
        xaxis = list(title = "Black-box prediction", range = axis_range),
        yaxis = list(title = "LIME surrogate prediction", range = axis_range),
        legend = list(orientation = "h", x = 0, y = -0.18),
        margin = list(l = 70, r = 30, b = 80, t = 55)
      )
  })

  output$lime_table <- renderTable(
    {
      lime_df <- lime_result()$contributions
      lime_df <- lime_df[order(lime_df$abs_contribution, decreasing = TRUE), ]
      data.frame(
        Feature = lime_df$label,
        `Selected value` = fmt(lime_df$selected_value, 1),
        `Local center` = fmt(lime_df$local_center, 1),
        `Local slope` = signed_fmt(lime_df$local_slope),
        Contribution = signed_fmt(lime_df$contribution),
        check.names = FALSE
      )
    },
    striped = TRUE,
    bordered = TRUE,
    spacing = "s"
  )

  output$compare_contributions <- renderPlotly({
    shap_df <- shap_result()$values[, c("feature", "label", "shap")]
    names(shap_df)[names(shap_df) == "shap"] <- "contribution"
    shap_df$method <- "SHAP"

    lime_df <- lime_result()$contributions[, c("feature", "label", "contribution")]
    lime_df$method <- "LIME"

    compare_df <- rbind(shap_df, lime_df)
    compare_df$label <- factor(compare_df$label, levels = rev(unname(feature_labels[predictor_names])))

    hover_text <- paste0(
      "<b>", compare_df$method, "</b><br>",
      compare_df$label, "<br>",
      "Contribution: ", signed_fmt(compare_df$contribution)
    )

    plotly::plot_ly(
      data = compare_df,
      type = "bar",
      orientation = "h",
      x = ~contribution,
      y = ~label,
      color = ~method,
      colors = c("SHAP" = "#2563eb", "LIME" = "#f97316"),
      text = hover_text,
      hoverinfo = "text"
    ) |>
      plotly::layout(
        title = "Feature contributions side by side",
        barmode = "group",
        xaxis = list(title = "Contribution", zeroline = TRUE),
        yaxis = list(title = ""),
        legend = list(orientation = "h", x = 0, y = -0.16),
        margin = list(l = 150, r = 30, b = 80, t = 55)
      )
  })

  output$compare_predictions <- renderPlotly({
    shap_res <- shap_result()
    lime_res <- lime_result()
    prediction_df <- data.frame(
      Method = c("Black-box", "SHAP sum", "LIME surrogate"),
      Value = c(shap_res$prediction, shap_res$reconstruction, lime_res$surrogate_prediction),
      Meaning = c(
        "The model prediction for the selected profile",
        "Average baseline plus all SHAP values",
        "Prediction from LIME's local weighted linear model"
      ),
      stringsAsFactors = FALSE
    )
    prediction_df$Method <- factor(prediction_df$Method, levels = prediction_df$Method)

    hover_text <- paste0(
      "<b>", prediction_df$Method, "</b><br>",
      "Value: ", fmt(prediction_df$Value), "<br>",
      prediction_df$Meaning
    )

    plotly::plot_ly(
      data = prediction_df,
      type = "bar",
      x = ~Method,
      y = ~Value,
      text = hover_text,
      hoverinfo = "text",
      marker = list(color = c("#0f172a", "#2563eb", "#f97316"))
    ) |>
      plotly::layout(
        title = "Prediction reconstruction",
        xaxis = list(title = ""),
        yaxis = list(title = "Predicted rating"),
        showlegend = FALSE,
        margin = list(l = 70, r = 30, b = 80, t = 55)
      )
  })
}
View(attitude)
shinyApp(ui, server)
