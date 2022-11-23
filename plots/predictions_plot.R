# load packages
library(tidyverse)
library(nbastatR)
library(gt)
library(gtExtras)
library(ggflags)

df <- read.csv("merged_1668598471_ep_2_wc_4_100000.csv")
get_progression_probabiltiies <- function(df, n_sim, metric = "reached") {
  df <- df[order(df$W, decreasing = TRUE),]
  prog_df <- data.frame(matrix(nrow=0, ncol = 5))
  for (i in 1:nrow(df)) {
    if (metric == "reached") {
      progression <- c(sum(df[i, c("R16", "QF", "SF", "RU", "W")]) / n_sim,
                       sum(df[i, c("QF", "SF", "RU", "W")]) / n_sim,
                       sum(df[i, c("SF", "RU", "W")]) / n_sim,
                       sum(df[i, c("RU", "W")]) / n_sim,
                       sum(df[i, c("W")]) / n_sim)
    } else if (metric == "knocked_out_at") {
      progression <- df[i,c("G", "R16", "QF", "SF", "RU", "W")] / n_sim
    }
    prog_df <- rbind(prog_df, progression)
  }
  prog_df <- cbind(df$team, prog_df)
  colnames(prog_df) <- c("team", "R16", "QF", "SF", "F", "W")
  return(prog_df)
}
prog_df <- get_progression_probabiltiies(df, 100000)
knocked_out_df <- get_progression_probabiltiies(df, 100000, "knocked_out_at")

# top <- prog_df[1:16,] %>%
#   mutate('logo' = paste0('flags/', team, '.png')) %>%
#   select(team, logo, everything()) %>%
#   gt() %>%
#   tab_header(
#     title = md("**World Cup 2022**"),
#     subtitle = "") %>% 
#   cols_label(team = "",
#              logo = "") %>%
#   cols_width(team ~ 80, 
#              logo ~ 50,
#              everything() ~ 60) %>% 
#   fmt_percent(columns = c(R16, QF, SF, F, W), 
#              decimals = 1) %>% 
#   data_color(
#     columns = c(R16, QF, SF, F, W),
#     colors = scales::col_numeric(
#       palette = paletteer::paletteer_d(
#         palette = "Redmonder::dPBIRdGn",
#         direction = 1
#       ) %>% as.character(),
#       domain = c(-0.5, 1), 
#       na.color = "#005C55FF"
#     )) %>%
#   text_transform(
#     locations = cells_body(columns = "logo"), 
#     fn = function(x) map_chr(x, ~{
#       local_image(filename =  as.character(.x), height = 30)
#     })
#   ) %>%
#   tab_options(
#     heading.title.font.size = 18,
#     heading.subtitle.font.size = 12,
#     heading.title.font.weight = 'bold',
#     column_labels.font.size = 12,
#     column_labels.font.weight = 'bold',
#     table.font.size = 11,
#     source_notes.font.size = 8,
#     data_row.padding = px(.8)
#   ) %>%
#   tab_source_note(
#     source_note = md("The Alan Turing Institute (Nick Barlow, Jack Roberts, Ryan Chan)<br>Based on 100,000 simulations<br>Data: GitHub (martj42/international_results)<br>Country Images: Flaticon.com and GitHub (lbenz730/world_cup_2022)")
#   )
# top
# 
# bottom <- prog_df[17:32,] %>%
#   mutate('logo' = paste0('flags/', team, '.png')) %>%
#   select(team, logo, everything()) %>%
#   gt() %>%
#   tab_header(
#     title = md("**World Cup 2022**"),
#     subtitle = "") %>% 
#   cols_label(team = "",
#              logo = "") %>%
#   cols_width(team ~ 80, 
#              logo ~ 50,
#              everything() ~ 60) %>% 
#   fmt_percent(columns = c(R16, QF, SF, F, W), 
#               decimals = 1) %>% 
#   data_color(
#     columns = c(R16, QF, SF, F, W),
#     colors = scales::col_numeric(
#       palette = paletteer::paletteer_d(
#         palette = "Redmonder::dPBIRdGn",
#         direction = 1
#       ) %>% as.character(),
#       domain = c(-0.5, 1), 
#       na.color = "#005C55FF"
#     )) %>%
#   text_transform(
#     locations = cells_body(columns = "logo"), 
#     fn = function(x) map_chr(x, ~{
#       local_image(filename =  as.character(.x), height = 30)
#     })
#   ) %>%
#   tab_options(
#     heading.title.font.size = 18,
#     heading.subtitle.font.size = 12,
#     heading.title.font.weight = 'bold',
#     column_labels.font.size = 12,
#     column_labels.font.weight = 'bold',
#     table.font.size = 11,
#     source_notes.font.size = 8,
#     data_row.padding = px(.8)
#   )
# bottom

# two_tables <- list(top, bottom)
# gt_two_column_layout(tables = two_tables, 
#                      output = 'save', 
#                      filename = 'predictions_two_cols.png', 
#                      vwidth = 825, 
#                      vheight = 475)

full_table <- prog_df %>%
  mutate('logo' = paste0('flags/', team, '.png')) %>%
  select(team, logo, everything()) %>%
  gt() %>%
  tab_header(
    title = md("**World Cup 2022**"),
    subtitle = "") %>% 
  cols_label(team = "",
             logo = "") %>%
  cols_width(team ~ 80, 
             logo ~ 50,
             everything() ~ 60) %>% 
  fmt_percent(columns = c(R16, QF, SF, F, W), 
              decimals = 1) %>% 
  data_color(
    columns = c(R16, QF, SF, F, W),
    colors = scales::col_numeric(
      palette = paletteer::paletteer_d(
        palette = "Redmonder::dPBIRdGn",
        direction = 1
      ) %>% as.character(),
      domain = c(-0.5, 1), 
      na.color = "#005C55FF"
    )) %>%
  text_transform(
    locations = cells_body(columns = "logo"), 
    fn = function(x) map_chr(x, ~{
      local_image(filename =  as.character(.x), height = 30)
    })
  ) %>%
  tab_options(
    heading.title.font.size = 20,
    heading.subtitle.font.size = 14,
    heading.title.font.weight = 'bold',
    column_labels.font.size = 14,
    column_labels.font.weight = 'bold',
    table.font.size = 12,
    source_notes.font.size = 8,
    data_row.padding = px(.8)
  ) %>%
  tab_source_note(
    source_note = md("The Alan Turing Institute (Nick Barlow, Jack Roberts, Ryan Chan)<br>Based on 100,000 simulations<br>Data: GitHub (martj42/international_results)<br>Country Images: Flaticon.com and GitHub (lbenz730/world_cup_2022)")
  )
full_table
# gtsave(full_table, filename = 'predictions.png')
 