library(corrplot)

FILE <- "../A27_be_virsutiniu_isskirciu.csv"
OUT_DIR <- "../kor_rez"
FEATURES <- c("Log_X_Index", "Log_Y_Index", "Empty_Index", "Square_Index", "Length_of_Conveyer",
              "Steel_Plate_Thickness", "Edges_Index", "Orientation_Index", "LogOfAreas", "Luminosity_Index")

KLASĖS <- c("Bumps" = 0, "Other_Faults" = 1)

df <- read.csv(FILE)

duomenys <- df[, FEATURES]
duomenys$class <- KLASĖS[trimws(df$class)]

rho <- cor(duomenys, method = "spearman")

dir.create(OUT_DIR, showWarnings = FALSE)

png(file.path(OUT_DIR, "koreliacija_spearman.png"), width = 1600, height = 1600, res = 200)
corrplot(rho, method = "circle", type = "lower", addCoef.col = "black",
         tl.col = "black", number.cex = 0.6, diag = FALSE)
dev.off()
