library(ggplot2)

FILE <- "../A27_be_virsutiniu_isskirciu.csv"
OUT_DIR <- "../kor_rez/sklaidos_diagramos"
PAIRS <- list(c("LogOfAreas", "Log_Y_Index"), c("Log_Y_Index", "Orientation_Index"),
              c("Log_X_Index", "LogOfAreas"), c("Log_X_Index", "Empty_Index"),
              c("Log_X_Index", "Orientation_Index"))

df <- read.csv(FILE)
df$class <- trimws(df$class)

dir.create(OUT_DIR, showWarnings = FALSE)

for (pora in PAIRS) {
  scatterplot <- ggplot(df) +
    aes(x = .data[[pora[1]]], y = .data[[pora[2]]], color = class) +
    geom_point() +
    geom_smooth(method = lm, se = FALSE, color = "red") +
    scale_color_manual(values = c("Bumps" = "steelblue", "Other_Faults" = "darkorange"))
  ggsave(file.path(OUT_DIR, paste0(pora[1], "_", pora[2], ".png")), scatterplot, width = 7, height = 5)
}
