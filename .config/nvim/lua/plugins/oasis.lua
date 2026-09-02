return {
  {
    "uhs-robert/oasis.nvim",
    lazy = false,
    priority = 1000,
    config = function()
      require("oasis").setup({
        style = "abyss",
        light_style = "starlight",
        light_intensity = 1,
      })
    end,
  },

  {
    "LazyVim/LazyVim",
    opts = {
      colorscheme = "oasis",
    },
  },
}
