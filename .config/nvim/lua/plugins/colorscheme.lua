return {
  "uhs-robert/oasis.nvim",
  lazy = false,
  priority = 1000,

  config = function()
    local handle = io.popen("gsettings get org.gnome.desktop.interface color-scheme")
    local result = handle:read("*a")
    handle:close()

    if result:match("default") then
      vim.opt.background = "light"
    else
      vim.opt.background = "dark"
    end

    local is_light = vim.o.background == "light"

    require("oasis").setup({
      transparent = true,
      style = is_light and "starlight" or "abyss",

      dark_intensity = 5,
      light_intensity = 1,
      match_paren_bg = true,
    })
    vim.cmd.colorscheme("oasis")
  end,
}
