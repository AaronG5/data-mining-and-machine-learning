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
pos <- corrplot(rho, method = "circle", type = "lower",
                tl.col = "black", tl.cex = 1.3, cl.cex = 1.3, diag = FALSE)$corrPos
koef <- round(pos$corr, 2)
koef[koef == 0] <- 0
text(pos$x, pos$y, sprintf("%.2f", koef), cex = 0.9, font = 2,
     col = ifelse(abs(pos$corr) >= 0.6, "white", "black"))
dev.off()
