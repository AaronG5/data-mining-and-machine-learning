library(corrplot)

FILE <- "../A27_be_virsutiniu_isskirciu.csv"
OUT_DIR <- "../kor_rez"

KLASĖS <- c("Bumps" = 0, "Other_Faults" = 1)

df <- read.csv(FILE)

duomenys <- df[, setdiff(names(df), "class")]
duomenys$class <- KLASĖS[trimws(df$class)]

rho <- cor(duomenys, method = "spearman")

dir.create(OUT_DIR, showWarnings = FALSE)

png(file.path(OUT_DIR, "koreliacija_spearman_visi.png"), width = 3000, height = 3000, res = 200)
corrplot(rho, method = "circle", type = "lower", addCoef.col = "black",
         tl.col = "black", number.cex = 0.5, diag = FALSE)
dev.off()
