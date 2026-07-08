############################################################
# Individual prediction app for datasets::attitude
#
# Purpose
# - Tune two hypothetical department profiles with sliders
# - Compare predictions from:
#   1. ordinary least-squares linear regression
#   2. robust regression using MASS::rlm
# - Show interpretable variable relevance, model fit metrics,
#   residual diagnostics, and hoverable graphics
#
# Run with:
#   install.packages(c("shiny", "plotly", "MASS"))
#   shiny::runApp("attitude_lm_robust_prediction_app.R")
############################################################

required_packages <- c("shiny", "plotly", "MASS")
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
library(MASS)

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

fmt <- function(x, digits = 2) {
  formatC(x, format = "f", digits = digits)
}

signed_fmt <- function(x, digits = 2) {
  paste0(ifelse(x >= 0, "+", ""), fmt(x, digits))
}

clip_value <- function(x, lower, upper) {
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
outcome_mean <- mean(attitude_data[[target_name]])
outcome_sd <- sd(attitude_data[[target_name]])

model_formula <- rating ~ complaints + privileges + learning + raises + critical + advance

linear_fit <- lm(model_formula, data = attitude_data)
robust_fit <- MASS::rlm(
  model_formula,
  data = attitude_data,
  psi = MASS::psi.huber,
  maxit = 100
)

model_fits <- list(
  lm = linear_fit,
  robust = robust_fit
)

model_labels <- c(
  lm = "Linear regression",
  robust = "Robust regression"
)

model_choices <- setNames(names(model_labels), unname(model_labels))

model_colours <- c(
  "Linear regression" = "#2563eb",
  "Robust regression" = "#f97316"
)

case_colours <- c(
  "Case A" = "#0891b2",
  "Case B" = "#db2777"
)

default_case_a <- 10
default_case_b <- 20

department_choices <- setNames(
  seq_len(nrow(attitude_data)),
  paste0(attitude_data$department, " - observed rating ", attitude_data$rating)
)

predict_from_model <- function(fit_obj, new_data) {
  as.numeric(predict(fit_obj, newdata = as.data.frame(new_data[, predictor_names, drop = FALSE])))
}

make_case_df <- function(values_named) {
  case_df <- as.data.frame(as.list(values_named[predictor_names]))
  names(case_df) <- predictor_names
  case_df
}

case_hover_text <- function(case_df, case_label, include_prediction = TRUE) {
  base_text <- paste0(
    "<b>", case_label, "</b><br>",
    paste(
      paste0(feature_labels[predictor_names], ": ", fmt(as.numeric(case_df[1, predictor_names]), 1)),
      collapse = "<br>"
    )
  )

  if (include_prediction) {
    lm_pred <- predict_from_model(linear_fit, case_df)
    robust_pred <- predict_from_model(robust_fit, case_df)
    base_text <- paste0(
      base_text,
      "<br>Linear prediction: ", fmt(lm_pred),
      "<br>Robust prediction: ", fmt(robust_pred)
    )
  }

  base_text
}

training_predictions_lm <- predict_from_model(linear_fit, attitude_data)
training_predictions_robust <- predict_from_model(robust_fit, attitude_data)
robust_weights <- robust_fit$w
if (is.null(robust_weights)) {
  robust_weights <- rep(1, nrow(attitude_data))
}

row_hover_text <- vapply(
  seq_len(nrow(attitude_data)),
  function(row_idx) {
    paste0(
      "<b>", attitude_data$department[[row_idx]], "</b><br>",
      "Observed rating: ", attitude_data$rating[[row_idx]], "<br>",
      "Linear fitted: ", fmt(training_predictions_lm[[row_idx]]), "<br>",
      "Robust fitted: ", fmt(training_predictions_robust[[row_idx]]), "<br>",
      "Robust weight: ", fmt(robust_weights[[row_idx]], 3), "<br>",
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

calculate_fit_metrics <- function(fit_obj, model_key) {
  fitted_values <- predict_from_model(fit_obj, attitude_data)
  residual_values <- attitude_data[[target_name]] - fitted_values
  sse <- sum(residual_values^2)
  tss <- sum((attitude_data[[target_name]] - outcome_mean)^2)
  r_squared <- 1 - sse / tss
  n_obs <- nrow(attitude_data)
  n_predictors <- length(predictor_names)
  adj_r_squared <- 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - n_predictors - 1)
  rmse <- sqrt(mean(residual_values^2))
  mae <- mean(abs(residual_values))
  median_abs_error <- median(abs(residual_values))
  scale_value <- if (model_key == "lm") {
    summary(fit_obj)$sigma
  } else {
    fit_obj$s
  }

  data.frame(
    model_key = model_key,
    Model = unname(model_labels[[model_key]]),
    RMSE = rmse,
    MAE = mae,
    `Median absolute error` = median_abs_error,
    `R-squared or pseudo R-squared` = r_squared,
    `Adjusted R-squared or pseudo adjusted R-squared` = adj_r_squared,
    `Residual scale` = scale_value,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

metrics_df <- rbind(
  calculate_fit_metrics(linear_fit, "lm"),
  calculate_fit_metrics(robust_fit, "robust")
)

calculate_importance <- function(fit_obj, model_key) {
  coef_vec <- coef(fit_obj)
  raw_coef <- coef_vec[predictor_names]
  std_coef <- raw_coef * feature_sd[predictor_names] / outcome_sd

  p_values <- rep(NA_real_, length(predictor_names))
  names(p_values) <- predictor_names

  if (model_key == "lm") {
    lm_coef_table <- summary(fit_obj)$coefficients
    p_values <- lm_coef_table[predictor_names, "Pr(>|t|)"]
  }

  data.frame(
    model_key = model_key,
    Model = unname(model_labels[[model_key]]),
    feature = predictor_names,
    Feature = unname(feature_labels[predictor_names]),
    `Raw coefficient` = as.numeric(raw_coef[predictor_names]),
    `Standardized coefficient` = as.numeric(std_coef[predictor_names]),
    `Absolute standardized coefficient` = abs(as.numeric(std_coef[predictor_names])),
    `Effect of +10 points` = as.numeric(raw_coef[predictor_names]) * 10,
    `Linear-model p-value` = as.numeric(p_values[predictor_names]),
    Direction = ifelse(raw_coef[predictor_names] >= 0, "higher values predict higher rating", "higher values predict lower rating"),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

importance_df <- rbind(
  calculate_importance(linear_fit, "lm"),
  calculate_importance(robust_fit, "robust")
)

calculate_case_contributions <- function(fit_obj, model_key, case_df, case_name) {
  coef_vec <- coef(fit_obj)
  raw_coef <- coef_vec[predictor_names]
  baseline_at_mean <- unname(coef_vec[[1]] + sum(raw_coef[predictor_names] * feature_mean[predictor_names]))
  current_values <- as.numeric(case_df[1, predictor_names])
  names(current_values) <- predictor_names
  contributions <- raw_coef[predictor_names] * (current_values[predictor_names] - feature_mean[predictor_names])
  prediction_value <- baseline_at_mean + sum(contributions)

  contribution_df <- data.frame(
    model_key = model_key,
    Model = unname(model_labels[[model_key]]),
    Case = case_name,
    feature = predictor_names,
    Feature = unname(feature_labels[predictor_names]),
    `Current value` = as.numeric(current_values[predictor_names]),
    `Training mean` = as.numeric(feature_mean[predictor_names]),
    `Raw coefficient` = as.numeric(raw_coef[predictor_names]),
    Contribution = as.numeric(contributions[predictor_names]),
    `Absolute contribution` = abs(as.numeric(contributions[predictor_names])),
    Direction = ifelse(contributions[predictor_names] >= 0, "pushes prediction up", "pushes prediction down"),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  list(
    baseline = baseline_at_mean,
    prediction = prediction_value,
    contributions = contribution_df
  )
}

make_profile_table <- function(case_a_df, case_b_df) {
  data.frame(
    Feature = unname(feature_labels[predictor_names]),
    Variable = predictor_names,
    `Case A` = as.numeric(case_a_df[1, predictor_names]),
    `Case B` = as.numeric(case_b_df[1, predictor_names]),
    `Training mean` = as.numeric(feature_mean[predictor_names]),
    `Data min` = as.numeric(feature_min[predictor_names]),
    `Data max` = as.numeric(feature_max[predictor_names]),
    Meaning = unname(feature_help[predictor_names]),
    check.names = FALSE
  )
}

plotly_empty_message <- function(message_text) {
  fig <- plotly::plot_ly()
  fig <- plotly::layout(
    fig,
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
  fig
}

app_css <- "
body {
  background: #f8fafc;
  color: #0f172a;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.hero {
  background:
    radial-gradient(circle at 15% 20%, rgba(56, 189, 248, 0.28), transparent 27%),
    radial-gradient(circle at 85% 18%, rgba(249, 115, 22, 0.24), transparent 24%),
    linear-gradient(135deg, #0f172a 0%, #1d4ed8 54%, #0f766e 100%);
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
  max-width: 1150px;
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
  font-size: 29px;
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
.compact-buttons .btn {
  margin: 3px 3px 6px 0;
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
@media (max-width: 1150px) {
  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
@media (max-width: 720px) {
  .metric-grid {
    grid-template-columns: 1fr;
  }
}
"

ui <- fluidPage(
  tags$head(tags$style(HTML(app_css))),

  div(
    class = "hero",
    h1("Two-case prediction lab: linear vs robust regression"),
    p(
      "This app uses ",
      tags$code("datasets::attitude"),
      ", an employee-attitude data set often used for psychology and organizational-behaviour examples. ",
      "Tune two hypothetical department profiles with sliders, compare predictions from ordinary linear regression and robust regression, ",
      "and inspect variable relevance, individual contributions, and model fit. Hover over plots to see exact values."
    )
  ),

  sidebarLayout(
    sidebarPanel(
      width = 3,
      h3("1. Start from departments"),
      selectInput(
        "case_a_row",
        "Case A starts from",
        choices = department_choices,
        selected = default_case_a
      ),
      selectInput(
        "case_b_row",
        "Case B starts from",
        choices = department_choices,
        selected = default_case_b
      ),
      div(
        class = "compact-buttons",
        actionButton("case_a_average", "Set A to average"),
        actionButton("case_b_average", "Set B to average"),
        actionButton("copy_a_to_b", "Copy A to B")
      ),
      div(class = "mini-note", "Department selectors initialise the sliders. After that, move sliders to create custom profiles."),
      hr(),

      h3("2. Tune covariates"),
      tabsetPanel(
        id = "case_slider_tabs",
        tabPanel("Case A", uiOutput("case_a_sliders")),
        tabPanel("Case B", uiOutput("case_b_sliders"))
      ),
      hr(),

      h3("3. Plot options"),
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
      selectInput(
        "surface_model",
        "Prediction surface model",
        choices = model_choices,
        selected = "lm"
      ),
      selectInput(
        "surface_anchor",
        "Hold other covariates at",
        choices = c("Case A", "Case B", "Training mean"),
        selected = "Case A"
      ),
      checkboxInput("show_formula", "Show model formula", value = FALSE),
      hr(),

      h3("4. Contribution view"),
      selectInput("detail_case", "Detailed case", choices = c("Case A" = "a", "Case B" = "b"), selected = "a"),
      selectInput("detail_model", "Detailed model", choices = model_choices, selected = "lm"),
      selectInput("contrib_compare_model", "Compare case contributions for", choices = model_choices, selected = "lm")
    ),

    mainPanel(
      width = 9,
      uiOutput("metric_cards"),

      tabsetPanel(
        tabPanel(
          "Predictions",
          div(class = "explain-card", uiOutput("prediction_text")),
          fluidRow(
            column(width = 6, plotlyOutput("prediction_bar", height = "420px")),
            column(width = 6, plotlyOutput("profile_plot", height = "420px"))
          ),
          div(class = "soft-card", tableOutput("profile_table"))
        ),

        tabPanel(
          "Variable relevance",
          div(class = "explain-card", uiOutput("importance_text")),
          fluidRow(
            column(width = 6, plotlyOutput("importance_plot", height = "450px")),
            column(width = 6, plotlyOutput("contribution_waterfall", height = "450px"))
          ),
          fluidRow(
            column(width = 7, plotlyOutput("case_contribution_compare", height = "410px")),
            column(width = 5, div(class = "soft-card", tableOutput("coefficient_table")))
          )
        ),

        tabPanel(
          "Model fit",
          div(class = "explain-card", uiOutput("fit_text")),
          fluidRow(
            column(width = 6, plotlyOutput("observed_fitted_plot", height = "420px")),
            column(width = 6, plotlyOutput("residual_plot", height = "420px"))
          ),
          fluidRow(
            column(width = 6, plotlyOutput("robust_weight_plot", height = "380px")),
            column(width = 6, div(class = "soft-card", tableOutput("metrics_table")))
          )
        ),

        tabPanel(
          "Data map",
          div(class = "explain-card", uiOutput("data_map_text")),
          conditionalPanel(
            condition = "input.show_formula == true",
            div(class = "formula-box", textOutput("formula_text"))
          ),
          fluidRow(
            column(width = 6, plotlyOutput("data_scatter", height = "430px")),
            column(width = 6, plotlyOutput("prediction_surface", height = "430px"))
          )
        ),

        tabPanel(
          "Plain-English guide",
          div(
            class = "explain-card",
            h3("What the sliders do"),
            p("Each slider changes one survey score for Case A or Case B. The app immediately recalculates each model's predicted overall attitude rating."),
            h3("How to read the two models"),
            p(strong("Linear regression"), "fits the line or plane that minimises squared errors. Large residuals can have a lot of influence."),
            p(strong("Robust regression"), "uses a similar linear equation, but downweights departments that look unusual under the model. This often gives a model that is less pulled around by outliers."),
            h3("How to read variable relevance"),
            p(strong("Global relevance"), "uses standardized coefficients. A bigger absolute standardized coefficient means the model changes more when that variable moves by one standard deviation."),
            p(strong("Individual contribution"), "starts at the model prediction for an average department, then adds the effect of each slider value being above or below the training mean."),
            h3("How to read fit metrics"),
            p(strong("RMSE"), "penalises big errors strongly. ", strong("MAE"), "is the average absolute error. ", strong("R-squared"), "summarises how much variation in observed ratings is captured by predictions. For the robust model it is a prediction-based pseudo R-squared, because robust regression is fitted with a different loss function.")
          )
        )
      )
    )
  )
)

server <- function(input, output, session) {
  build_slider_ui <- function(case_prefix, selected_idx) {
    tagList(
      lapply(
        predictor_names,
        function(feature_name) {
          sliderInput(
            inputId = paste0(case_prefix, "_", feature_name),
            label = paste0(feature_labels[[feature_name]], " (", feature_name, ")"),
            min = floor(feature_min[[feature_name]]),
            max = ceiling(feature_max[[feature_name]]),
            value = attitude_data[[feature_name]][[selected_idx]],
            step = 1
          )
        }
      )
    )
  }

  update_case_sliders <- function(case_prefix, values_named) {
    for (feature_name in predictor_names) {
      updateSliderInput(
        session,
        inputId = paste0(case_prefix, "_", feature_name),
        value = round(values_named[[feature_name]])
      )
    }
  }

  output$case_a_sliders <- renderUI({
    selected_idx <- as.integer(input$case_a_row %||% default_case_a)
    build_slider_ui("a", selected_idx)
  })

  output$case_b_sliders <- renderUI({
    selected_idx <- as.integer(input$case_b_row %||% default_case_b)
    build_slider_ui("b", selected_idx)
  })

  observeEvent(input$case_a_row, {
    selected_idx <- as.integer(input$case_a_row)
    update_case_sliders("a", attitude_data[selected_idx, predictor_names])
  }, ignoreInit = TRUE)

  observeEvent(input$case_b_row, {
    selected_idx <- as.integer(input$case_b_row)
    update_case_sliders("b", attitude_data[selected_idx, predictor_names])
  }, ignoreInit = TRUE)

  observeEvent(input$case_a_average, {
    update_case_sliders("a", as.list(feature_mean[predictor_names]))
  })

  observeEvent(input$case_b_average, {
    update_case_sliders("b", as.list(feature_mean[predictor_names]))
  })

  observeEvent(input$copy_a_to_b, {
    update_case_sliders("b", as.list(as.numeric(case_a()[1, predictor_names]) |> setNames(predictor_names)))
  })

  case_a <- reactive({
    values_named <- vapply(
      predictor_names,
      function(feature_name) {
        input[[paste0("a_", feature_name)]] %||% attitude_data[[feature_name]][[default_case_a]]
      },
      numeric(1)
    )
    names(values_named) <- predictor_names
    make_case_df(values_named)
  })

  case_b <- reactive({
    values_named <- vapply(
      predictor_names,
      function(feature_name) {
        input[[paste0("b_", feature_name)]] %||% attitude_data[[feature_name]][[default_case_b]]
      },
      numeric(1)
    )
    names(values_named) <- predictor_names
    make_case_df(values_named)
  })

  prediction_df_reactive <- reactive({
    case_a_df <- case_a()
    case_b_df <- case_b()
    prediction_df <- data.frame(
      Case = rep(c("Case A", "Case B"), each = 2),
      model_key = rep(c("lm", "robust"), times = 2),
      Model = rep(unname(model_labels[c("lm", "robust")]), times = 2),
      Prediction = c(
        predict_from_model(linear_fit, case_a_df),
        predict_from_model(robust_fit, case_a_df),
        predict_from_model(linear_fit, case_b_df),
        predict_from_model(robust_fit, case_b_df)
      ),
      stringsAsFactors = FALSE
    )
    prediction_df
  })

  selected_case_df <- reactive({
    if (identical(input$detail_case, "a")) {
      case_a()
    } else {
      case_b()
    }
  })

  selected_case_label <- reactive({
    if (identical(input$detail_case, "a")) "Case A" else "Case B"
  })

  selected_model_fit <- reactive({
    model_fits[[input$detail_model]]
  })

  selected_model_label <- reactive({
    unname(model_labels[[input$detail_model]])
  })

  detail_contribution <- reactive({
    calculate_case_contributions(
      fit_obj = selected_model_fit(),
      model_key = input$detail_model,
      case_df = selected_case_df(),
      case_name = selected_case_label()
    )
  })

  all_contributions <- reactive({
    case_a_df <- case_a()
    case_b_df <- case_b()

    lm_a <- calculate_case_contributions(linear_fit, "lm", case_a_df, "Case A")$contributions
    lm_b <- calculate_case_contributions(linear_fit, "lm", case_b_df, "Case B")$contributions
    robust_a <- calculate_case_contributions(robust_fit, "robust", case_a_df, "Case A")$contributions
    robust_b <- calculate_case_contributions(robust_fit, "robust", case_b_df, "Case B")$contributions

    rbind(lm_a, lm_b, robust_a, robust_b)
  })

  output$metric_cards <- renderUI({
    pred_df <- prediction_df_reactive()
    lm_a <- pred_df$Prediction[pred_df$Case == "Case A" & pred_df$model_key == "lm"]
    lm_b <- pred_df$Prediction[pred_df$Case == "Case B" & pred_df$model_key == "lm"]
    robust_a <- pred_df$Prediction[pred_df$Case == "Case A" & pred_df$model_key == "robust"]
    robust_b <- pred_df$Prediction[pred_df$Case == "Case B" & pred_df$model_key == "robust"]

    div(
      class = "metric-grid",
      div(
        class = "metric-card",
        div(class = "metric-label", "Case A - linear"),
        div(class = "metric-value", fmt(lm_a)),
        div(class = "metric-note", "OLS prediction for the Case A slider profile.")
      ),
      div(
        class = "metric-card",
        div(class = "metric-label", "Case B - linear"),
        div(class = "metric-value", fmt(lm_b)),
        div(class = "metric-note", paste0("Case B minus Case A: ", signed_fmt(lm_b - lm_a), "."))
      ),
      div(
        class = "metric-card",
        div(class = "metric-label", "Case A - robust"),
        div(class = "metric-value", fmt(robust_a)),
        div(class = "metric-note", "Robust-regression prediction for Case A.")
      ),
      div(
        class = "metric-card",
        div(class = "metric-label", "Case B - robust"),
        div(class = "metric-value", fmt(robust_b)),
        div(class = "metric-note", paste0("Case B minus Case A: ", signed_fmt(robust_b - robust_a), "."))
      )
    )
  })

  output$prediction_text <- renderUI({
    pred_df <- prediction_df_reactive()
    lm_gap <- pred_df$Prediction[pred_df$Case == "Case B" & pred_df$model_key == "lm"] -
      pred_df$Prediction[pred_df$Case == "Case A" & pred_df$model_key == "lm"]
    robust_gap <- pred_df$Prediction[pred_df$Case == "Case B" & pred_df$model_key == "robust"] -
      pred_df$Prediction[pred_df$Case == "Case A" & pred_df$model_key == "robust"]

    HTML(paste0(
      "<b>Individual predictions:</b> the sliders define two department profiles. ",
      "The linear model currently predicts Case B to be <b>", signed_fmt(lm_gap),
      "</b> points relative to Case A. The robust model predicts Case B to be <b>",
      signed_fmt(robust_gap), "</b> points relative to Case A. ",
      "Hover over bars and profile lines to see the exact values."
    ))
  })

  output$importance_text <- renderUI({
    lm_top <- importance_df[importance_df$model_key == "lm", ]
    lm_top <- lm_top[which.max(lm_top$`Absolute standardized coefficient`), ]
    robust_top <- importance_df[importance_df$model_key == "robust", ]
    robust_top <- robust_top[which.max(robust_top$`Absolute standardized coefficient`), ]

    HTML(paste0(
      "<b>Variable relevance:</b> the left plot uses standardized coefficients, so variables can be compared on a common scale. ",
      "In the linear model, the largest standardized effect is <b>", lm_top$Feature,
      "</b>. In the robust model, the largest standardized effect is <b>", robust_top$Feature,
      "</b>. The waterfall explains one selected case by starting from an average profile and adding each covariate's contribution."
    ))
  })

  output$fit_text <- renderUI({
    lm_rmse <- metrics_df$RMSE[metrics_df$model_key == "lm"]
    robust_rmse <- metrics_df$RMSE[metrics_df$model_key == "robust"]
    lm_r2 <- metrics_df$`R-squared or pseudo R-squared`[metrics_df$model_key == "lm"]
    robust_r2 <- metrics_df$`R-squared or pseudo R-squared`[metrics_df$model_key == "robust"]

    HTML(paste0(
      "<b>Model fit:</b> the linear model has RMSE <b>", fmt(lm_rmse),
      "</b> and R-squared <b>", fmt(lm_r2, 3),
      "</b>. The robust model has RMSE <b>", fmt(robust_rmse),
      "</b> and prediction-based pseudo R-squared <b>", fmt(robust_r2, 3),
      "</b>. Robust regression may have slightly different fit metrics because it intentionally downweights unusual departments."
    ))
  })

  output$data_map_text <- renderUI({
    HTML(paste0(
      "<b>Data map:</b> choose two covariates in the sidebar. ",
      "The scatterplot shows the actual departments; the heatmap shows model predictions while the other covariates are held fixed at ",
      "<b>", input$surface_anchor, "</b>. Case A and Case B are overlaid as large symbols."
    ))
  })

  output$formula_text <- renderText({
    paste(deparse(model_formula), collapse = "\n")
  })

  output$profile_table <- renderTable({
    profile_table <- make_profile_table(case_a(), case_b())
    profile_table$`Case A` <- fmt(profile_table$`Case A`, 1)
    profile_table$`Case B` <- fmt(profile_table$`Case B`, 1)
    profile_table$`Training mean` <- fmt(profile_table$`Training mean`, 1)
    profile_table$`Data min` <- fmt(profile_table$`Data min`, 1)
    profile_table$`Data max` <- fmt(profile_table$`Data max`, 1)
    profile_table
  }, striped = TRUE, bordered = TRUE, spacing = "s")

  output$prediction_bar <- renderPlotly({
    pred_df <- prediction_df_reactive()
    pred_df$Hover <- paste0(
      "<b>", pred_df$Case, "</b><br>",
      "Model: ", pred_df$Model, "<br>",
      "Predicted rating: ", fmt(pred_df$Prediction)
    )

    fig <- plotly::plot_ly(
      data = pred_df,
      x = ~Case,
      y = ~Prediction,
      color = ~Model,
      colors = model_colours,
      type = "bar",
      text = ~Hover,
      hoverinfo = "text"
    )
    fig <- plotly::layout(
      fig,
      title = "Predicted rating for two tuned profiles",
      barmode = "group",
      xaxis = list(title = ""),
      yaxis = list(title = "Predicted overall rating"),
      legend = list(orientation = "h", x = 0, y = -0.16),
      margin = list(l = 70, r = 30, b = 80, t = 55),
      shapes = list(
        list(
          type = "line",
          x0 = -0.4,
          x1 = 1.4,
          y0 = outcome_mean,
          y1 = outcome_mean,
          line = list(color = "#64748b", width = 2, dash = "dot")
        )
      ),
      annotations = list(
        list(
          x = 1.4,
          y = outcome_mean,
          text = paste0("observed mean = ", fmt(outcome_mean)),
          showarrow = FALSE,
          xanchor = "right",
          yanchor = "bottom",
          font = list(color = "#64748b")
        )
      )
    )
    fig
  })

  output$profile_plot <- renderPlotly({
    case_a_df <- case_a()
    case_b_df <- case_b()
    plot_df <- rbind(
      data.frame(Case = "Case A", Feature = unname(feature_labels[predictor_names]), Value = as.numeric(case_a_df[1, predictor_names]), stringsAsFactors = FALSE),
      data.frame(Case = "Case B", Feature = unname(feature_labels[predictor_names]), Value = as.numeric(case_b_df[1, predictor_names]), stringsAsFactors = FALSE),
      data.frame(Case = "Training mean", Feature = unname(feature_labels[predictor_names]), Value = as.numeric(feature_mean[predictor_names]), stringsAsFactors = FALSE)
    )
    plot_df$Feature <- factor(plot_df$Feature, levels = unname(feature_labels[predictor_names]))
    plot_df$Hover <- paste0(
      "<b>", plot_df$Case, "</b><br>",
      plot_df$Feature, ": ", fmt(plot_df$Value, 1)
    )

    fig <- plotly::plot_ly(
      data = plot_df,
      x = ~Feature,
      y = ~Value,
      color = ~Case,
      colors = c("Case A" = "#0891b2", "Case B" = "#db2777", "Training mean" = "#64748b"),
      type = "scatter",
      mode = "lines+markers",
      text = ~Hover,
      hoverinfo = "text",
      line = list(width = 3),
      marker = list(size = 9)
    )
    fig <- plotly::layout(
      fig,
      title = "Covariate profiles",
      xaxis = list(title = "", tickangle = -35),
      yaxis = list(title = "Survey score"),
      legend = list(orientation = "h", x = 0, y = -0.28),
      margin = list(l = 70, r = 30, b = 120, t = 55)
    )
    fig
  })

  output$importance_plot <- renderPlotly({
    imp_df <- importance_df
    imp_df$Feature <- factor(imp_df$Feature, levels = rev(unname(feature_labels[predictor_names])))
    imp_df$Hover <- paste0(
      "<b>", imp_df$Feature, "</b><br>",
      "Model: ", imp_df$Model, "<br>",
      "Raw coefficient: ", signed_fmt(imp_df$`Raw coefficient`, 3), "<br>",
      "Standardized coefficient: ", signed_fmt(imp_df$`Standardized coefficient`, 3), "<br>",
      "Effect of +10 points: ", signed_fmt(imp_df$`Effect of +10 points`), "<br>",
      imp_df$Direction,
      ifelse(is.na(imp_df$`Linear-model p-value`), "", paste0("<br>Linear-model p-value: ", fmt(imp_df$`Linear-model p-value`, 4)))
    )

    fig <- plotly::plot_ly(
      data = imp_df,
      x = ~`Standardized coefficient`,
      y = ~Feature,
      color = ~Model,
      colors = model_colours,
      type = "bar",
      orientation = "h",
      text = ~Hover,
      hoverinfo = "text"
    )
    fig <- plotly::layout(
      fig,
      title = "Global variable relevance: standardized coefficients",
      barmode = "group",
      xaxis = list(title = "Standardized coefficient", zeroline = TRUE),
      yaxis = list(title = ""),
      legend = list(orientation = "h", x = 0, y = -0.16),
      margin = list(l = 160, r = 30, b = 80, t = 55)
    )
    fig
  })

  output$contribution_waterfall <- renderPlotly({
    contribution_obj <- detail_contribution()
    contribution_df <- contribution_obj$contributions
    contribution_df <- contribution_df[order(contribution_df$`Absolute contribution`, decreasing = TRUE), ]

    x_labels <- c("Average profile", contribution_df$Feature, "Prediction")
    y_values <- c(contribution_obj$baseline, contribution_df$Contribution, 0)
    measures <- c("absolute", rep("relative", nrow(contribution_df)), "total")
    hover_text <- c(
      paste0("<b>Average profile baseline</b><br>", selected_model_label(), " prediction at training means: ", fmt(contribution_obj$baseline)),
      paste0(
        "<b>", contribution_df$Feature, "</b><br>",
        "Current value: ", fmt(contribution_df$`Current value`, 1), "<br>",
        "Training mean: ", fmt(contribution_df$`Training mean`, 1), "<br>",
        "Coefficient: ", signed_fmt(contribution_df$`Raw coefficient`, 3), "<br>",
        "Contribution: ", signed_fmt(contribution_df$Contribution), "<br>",
        contribution_df$Direction
      ),
      paste0("<b>", selected_case_label(), " prediction</b><br>", selected_model_label(), ": ", fmt(contribution_obj$prediction))
    )

    fig <- plotly::plot_ly(
      type = "waterfall",
      x = x_labels,
      y = y_values,
      measure = measures,
      text = c(fmt(contribution_obj$baseline), signed_fmt(contribution_df$Contribution), fmt(contribution_obj$prediction)),
      textposition = "outside",
      hovertext = hover_text,
      hoverinfo = "text",
      increasing = list(marker = list(color = "#16a34a")),
      decreasing = list(marker = list(color = "#dc2626")),
      totals = list(marker = list(color = ifelse(input$detail_model == "lm", "#2563eb", "#f97316"))),
      connector = list(line = list(color = "#94a3b8"))
    )
    fig <- plotly::layout(
      fig,
      title = paste0(selected_case_label(), " contribution waterfall - ", selected_model_label()),
      yaxis = list(title = "Predicted rating"),
      xaxis = list(title = ""),
      margin = list(l = 70, r = 30, b = 110, t = 55)
    )
    fig
  })

  output$case_contribution_compare <- renderPlotly({
    contribution_df <- all_contributions()
    contribution_df <- contribution_df[contribution_df$model_key == input$contrib_compare_model, ]
    contribution_df$Feature <- factor(contribution_df$Feature, levels = rev(unname(feature_labels[predictor_names])))
    contribution_df$Hover <- paste0(
      "<b>", contribution_df$Case, "</b><br>",
      "Model: ", contribution_df$Model, "<br>",
      contribution_df$Feature, "<br>",
      "Current value: ", fmt(contribution_df$`Current value`, 1), "<br>",
      "Training mean: ", fmt(contribution_df$`Training mean`, 1), "<br>",
      "Contribution: ", signed_fmt(contribution_df$Contribution)
    )

    fig <- plotly::plot_ly(
      data = contribution_df,
      x = ~Contribution,
      y = ~Feature,
      color = ~Case,
      colors = case_colours,
      type = "bar",
      orientation = "h",
      text = ~Hover,
      hoverinfo = "text"
    )
    fig <- plotly::layout(
      fig,
      title = paste0("Case-specific contributions - ", unname(model_labels[[input$contrib_compare_model]])),
      barmode = "group",
      xaxis = list(title = "Contribution relative to average profile", zeroline = TRUE),
      yaxis = list(title = ""),
      legend = list(orientation = "h", x = 0, y = -0.16),
      margin = list(l = 160, r = 30, b = 80, t = 55)
    )
    fig
  })

  output$coefficient_table <- renderTable({
    table_df <- importance_df
    table_df <- table_df[order(table_df$Model, -table_df$`Absolute standardized coefficient`), ]
    data.frame(
      Model = table_df$Model,
      Feature = table_df$Feature,
      `Raw coef.` = signed_fmt(table_df$`Raw coefficient`, 3),
      `Std. coef.` = signed_fmt(table_df$`Standardized coefficient`, 3),
      `+10 point effect` = signed_fmt(table_df$`Effect of +10 points`),
      `LM p-value` = ifelse(is.na(table_df$`Linear-model p-value`), "not used", fmt(table_df$`Linear-model p-value`, 4)),
      check.names = FALSE
    )
  }, striped = TRUE, bordered = TRUE, spacing = "s")

  output$observed_fitted_plot <- renderPlotly({
    fit_df <- rbind(
      data.frame(
        department = attitude_data$department,
        Model = "Linear regression",
        Observed = attitude_data$rating,
        Fitted = training_predictions_lm,
        Residual = attitude_data$rating - training_predictions_lm,
        RobustWeight = NA_real_,
        stringsAsFactors = FALSE
      ),
      data.frame(
        department = attitude_data$department,
        Model = "Robust regression",
        Observed = attitude_data$rating,
        Fitted = training_predictions_robust,
        Residual = attitude_data$rating - training_predictions_robust,
        RobustWeight = robust_weights,
        stringsAsFactors = FALSE
      )
    )
    fit_df$Hover <- paste0(
      "<b>", fit_df$department, "</b><br>",
      "Model: ", fit_df$Model, "<br>",
      "Observed rating: ", fmt(fit_df$Observed), "<br>",
      "Fitted rating: ", fmt(fit_df$Fitted), "<br>",
      "Residual: ", signed_fmt(fit_df$Residual),
      ifelse(is.na(fit_df$RobustWeight), "", paste0("<br>Robust weight: ", fmt(fit_df$RobustWeight, 3)))
    )
    axis_range <- range(c(fit_df$Observed, fit_df$Fitted), finite = TRUE)
    pad_val <- max(1, diff(axis_range) * 0.08)
    axis_range <- c(axis_range[1] - pad_val, axis_range[2] + pad_val)

    fig <- plotly::plot_ly(
      data = fit_df,
      x = ~Observed,
      y = ~Fitted,
      color = ~Model,
      colors = model_colours,
      type = "scatter",
      mode = "markers",
      text = ~Hover,
      hoverinfo = "text",
      marker = list(size = 11, opacity = 0.78, line = list(color = "#0f172a", width = 1))
    )
    fig <- plotly::add_lines(
      fig,
      x = axis_range,
      y = axis_range,
      line = list(color = "#dc2626", dash = "dash", width = 2),
      name = "Perfect fit",
      inherit = FALSE
    )
    fig <- plotly::layout(
      fig,
      title = "Observed vs fitted ratings",
      xaxis = list(title = "Observed rating", range = axis_range),
      yaxis = list(title = "Fitted rating", range = axis_range),
      legend = list(orientation = "h", x = 0, y = -0.16),
      margin = list(l = 70, r = 30, b = 80, t = 55)
    )
    fig
  })

  output$residual_plot <- renderPlotly({
    residual_df <- rbind(
      data.frame(
        department = attitude_data$department,
        Model = "Linear regression",
        Fitted = training_predictions_lm,
        Residual = attitude_data$rating - training_predictions_lm,
        RobustWeight = NA_real_,
        stringsAsFactors = FALSE
      ),
      data.frame(
        department = attitude_data$department,
        Model = "Robust regression",
        Fitted = training_predictions_robust,
        Residual = attitude_data$rating - training_predictions_robust,
        RobustWeight = robust_weights,
        stringsAsFactors = FALSE
      )
    )
    residual_df$Hover <- paste0(
      "<b>", residual_df$department, "</b><br>",
      "Model: ", residual_df$Model, "<br>",
      "Fitted rating: ", fmt(residual_df$Fitted), "<br>",
      "Residual: ", signed_fmt(residual_df$Residual),
      ifelse(is.na(residual_df$RobustWeight), "", paste0("<br>Robust weight: ", fmt(residual_df$RobustWeight, 3)))
    )

    fig <- plotly::plot_ly(
      data = residual_df,
      x = ~Fitted,
      y = ~Residual,
      color = ~Model,
      colors = model_colours,
      type = "scatter",
      mode = "markers",
      text = ~Hover,
      hoverinfo = "text",
      marker = list(size = 11, opacity = 0.78, line = list(color = "#0f172a", width = 1))
    )
    fig <- plotly::layout(
      fig,
      title = "Residual diagnostics",
      xaxis = list(title = "Fitted rating"),
      yaxis = list(title = "Residual: observed minus fitted", zeroline = TRUE),
      legend = list(orientation = "h", x = 0, y = -0.16),
      margin = list(l = 70, r = 30, b = 80, t = 55),
      shapes = list(
        list(
          type = "line",
          xref = "paper",
          x0 = 0,
          x1 = 1,
          y0 = 0,
          y1 = 0,
          line = list(color = "#64748b", width = 2, dash = "dot")
        )
      )
    )
    fig
  })

  output$robust_weight_plot <- renderPlotly({
    weight_df <- data.frame(
      department = attitude_data$department,
      Observed = attitude_data$rating,
      RobustFitted = training_predictions_robust,
      RobustResidual = attitude_data$rating - training_predictions_robust,
      RobustWeight = robust_weights,
      stringsAsFactors = FALSE
    )
    weight_df <- weight_df[order(weight_df$RobustWeight), ]
    weight_df$department <- factor(weight_df$department, levels = weight_df$department)
    weight_df$Hover <- paste0(
      "<b>", weight_df$department, "</b><br>",
      "Observed rating: ", fmt(weight_df$Observed), "<br>",
      "Robust fitted: ", fmt(weight_df$RobustFitted), "<br>",
      "Robust residual: ", signed_fmt(weight_df$RobustResidual), "<br>",
      "Robust weight: ", fmt(weight_df$RobustWeight, 3)
    )

    fig <- plotly::plot_ly(
      data = weight_df,
      x = ~department,
      y = ~RobustWeight,
      type = "bar",
      text = ~Hover,
      hoverinfo = "text",
      marker = list(color = ifelse(weight_df$RobustWeight < 0.95, "#f97316", "#2563eb"))
    )
    fig <- plotly::layout(
      fig,
      title = "Robust-regression case weights",
      xaxis = list(title = "", tickangle = -60),
      yaxis = list(title = "Final robust weight", range = c(0, 1.05)),
      margin = list(l = 70, r = 30, b = 110, t = 55)
    )
    fig
  })

  output$metrics_table <- renderTable({
    table_df <- metrics_df
    data.frame(
      Model = table_df$Model,
      RMSE = fmt(table_df$RMSE),
      MAE = fmt(table_df$MAE),
      `Median abs. error` = fmt(table_df$`Median absolute error`),
      `R-squared / pseudo` = fmt(table_df$`R-squared or pseudo R-squared`, 3),
      `Adjusted / pseudo adjusted` = fmt(table_df$`Adjusted R-squared or pseudo adjusted R-squared`, 3),
      `Residual scale` = fmt(table_df$`Residual scale`),
      check.names = FALSE
    )
  }, striped = TRUE, bordered = TRUE, spacing = "s")

  output$data_scatter <- renderPlotly({
    x_feature <- input$plot_x
    y_feature <- input$plot_y

    if (identical(x_feature, y_feature)) {
      return(plotly_empty_message("Choose two different features for the X and Y axes."))
    }

    case_a_df <- case_a()
    case_b_df <- case_b()

    fig <- plotly::plot_ly(
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
    )
    fig <- plotly::add_markers(
      fig,
      x = case_a_df[[x_feature]],
      y = case_a_df[[y_feature]],
      text = case_hover_text(case_a_df, "Case A"),
      hoverinfo = "text",
      marker = list(size = 22, color = "#0891b2", symbol = "star", line = list(color = "#111827", width = 2)),
      name = "Case A",
      inherit = FALSE
    )
    fig <- plotly::add_markers(
      fig,
      x = case_b_df[[x_feature]],
      y = case_b_df[[y_feature]],
      text = case_hover_text(case_b_df, "Case B"),
      hoverinfo = "text",
      marker = list(size = 20, color = "#db2777", symbol = "diamond", line = list(color = "#111827", width = 2)),
      name = "Case B",
      inherit = FALSE
    )
    fig <- plotly::layout(
      fig,
      title = "Observed departments plus tuned cases",
      xaxis = list(title = feature_labels[[x_feature]]),
      yaxis = list(title = feature_labels[[y_feature]]),
      legend = list(orientation = "h", x = 0, y = -0.18),
      margin = list(l = 70, r = 30, b = 80, t = 55)
    )
    fig
  })

  output$prediction_surface <- renderPlotly({
    x_feature <- input$plot_x
    y_feature <- input$plot_y

    if (identical(x_feature, y_feature)) {
      return(plotly_empty_message("Choose two different features for the X and Y axes."))
    }

    case_a_df <- case_a()
    case_b_df <- case_b()

    anchor_values <- if (identical(input$surface_anchor, "Case A")) {
      case_a_df
    } else if (identical(input$surface_anchor, "Case B")) {
      case_b_df
    } else {
      make_case_df(feature_mean[predictor_names])
    }

    surface_fit <- model_fits[[input$surface_model]]
    surface_label <- unname(model_labels[[input$surface_model]])

    x_seq <- seq(feature_min[[x_feature]], feature_max[[x_feature]], length.out = 55)
    y_seq <- seq(feature_min[[y_feature]], feature_max[[y_feature]], length.out = 55)
    grid_df <- expand.grid(x_value = x_seq, y_value = y_seq)
    prediction_grid <- anchor_values[rep(1, nrow(grid_df)), predictor_names, drop = FALSE]
    prediction_grid[[x_feature]] <- grid_df$x_value
    prediction_grid[[y_feature]] <- grid_df$y_value
    grid_prediction <- predict_from_model(surface_fit, prediction_grid)
    z_matrix <- matrix(grid_prediction, nrow = length(x_seq), ncol = length(y_seq))

    hover_template <- paste0(
      feature_labels[[x_feature]], ": %{x:.1f}<br>",
      feature_labels[[y_feature]], ": %{y:.1f}<br>",
      surface_label, " prediction: %{z:.2f}<extra></extra>"
    )

    fig <- plotly::plot_ly()
    fig <- plotly::add_heatmap(
      fig,
      x = x_seq,
      y = y_seq,
      z = t(z_matrix),
      colors = c("#eff6ff", "#93c5fd", "#2563eb", "#1e3a8a", "#0f172a"),
      hovertemplate = hover_template,
      colorbar = list(title = "Predicted<br>rating"),
      name = "Prediction surface"
    )
    fig <- plotly::add_markers(
      fig,
      x = attitude_data[[x_feature]],
      y = attitude_data[[y_feature]],
      text = row_hover_text,
      hoverinfo = "text",
      marker = list(size = 7, color = "white", opacity = 0.70, line = list(color = "#0f172a", width = 1)),
      name = "Observed departments",
      inherit = FALSE
    )
    fig <- plotly::add_markers(
      fig,
      x = case_a_df[[x_feature]],
      y = case_a_df[[y_feature]],
      text = case_hover_text(case_a_df, "Case A"),
      hoverinfo = "text",
      marker = list(size = 22, color = "#0891b2", symbol = "star", line = list(color = "#111827", width = 2)),
      name = "Case A",
      inherit = FALSE
    )
    fig <- plotly::add_markers(
      fig,
      x = case_b_df[[x_feature]],
      y = case_b_df[[y_feature]],
      text = case_hover_text(case_b_df, "Case B"),
      hoverinfo = "text",
      marker = list(size = 20, color = "#db2777", symbol = "diamond", line = list(color = "#111827", width = 2)),
      name = "Case B",
      inherit = FALSE
    )
    fig <- plotly::layout(
      fig,
      title = paste0("Two-feature prediction surface - ", surface_label),
      xaxis = list(title = feature_labels[[x_feature]]),
      yaxis = list(title = feature_labels[[y_feature]]),
      legend = list(orientation = "h", x = 0, y = -0.18),
      margin = list(l = 70, r = 30, b = 80, t = 55)
    )
    fig
  })
}

shinyApp(ui, server)
