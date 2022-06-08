library(shiny)
library(reticulate)  # loads python interpreter into R environment
use_python('/Users/asoa/miniconda3/envs/malaria_class/bin/python3')  # change to your python executable
source_python('predict_fn.py')

# Define UI for application that accepts a blood stain image and returns the classifcation result
ui <- fluidPage(

    # Application title
    titlePanel("Malaria Classification"),

    # File input widget 
    sidebarLayout(
        sidebarPanel(
            fileInput(inputId = 'image_upload',
                        label = 'Target Image',
                        multiple = TRUE,
                        accept = c('image/png'))
        ),

        # Display the prediction results and show the image
        mainPanel(
           verbatimTextOutput(outputId = 'pred_result'),
           imageOutput(outputId = 'img_output', width="500px", height="500px")
        )
    )
)

# get prediction from python environment; process_image processes image and calls cnn_predict for classification
get_prediction <- function(image) {
  pred <- process_image(image)
  return(pred)
}

server <- function(input, output, session) {
    # helper function to remove forward slashes in file name
    files <- reactive({
      files <- input$image_upload
      files$datapath <- gsub("\\\\", "/", files$datapath)
      files
    })
    # event listener that fires when image is uploaded
    observeEvent(input$image_upload, {
      in_file <- input$image_upload
      if(is.null(in_file))
        return(list(src=""))
      # file.copy(in_file$datapath, file.path('.', in_file$name))
      pred <- get_prediction(in_file$datapath)
      if(pred == 'pos') {
        # reactive binding to render prediction result
        output$pred_result <- renderPrint("Positive")
      } else {
        output$pred_result <- renderPrint("Negative")
      }
      # reactive binding to render image
      output$img_output <- renderImage({
        list(src=in_file$datapath)}, deleteFile = T)
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
