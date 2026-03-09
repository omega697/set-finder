package com.guywithburrito.setfinder.ml

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer

/**
 * Interface for the attribute expert model.
 * Produces raw head outputs (FloatArrays) from a preprocessed image buffer.
 */
interface CardExpertModel {
    /**
     * Executes inference on the provided ByteBuffer (preprocessed image).
     * Returns a map of head index to raw FloatArray probabilities.
     */
    fun predict(buffer: ByteBuffer): Map<Int, FloatArray>
    
    fun close()
}

/**
 * TFLite implementation of the CardExpertModel.
 */
class TFLiteExpertModel(
    context: Context, 
    modelPath: String = "set_card_model_final.tflite"
) : CardExpertModel {
    private val interpreter: Interpreter = Interpreter(FileUtil.loadMappedFile(context, modelPath))

    override fun predict(buffer: ByteBuffer): Map<Int, FloatArray> {
        buffer.rewind()
        val outputs = mutableMapOf<Int, Any>()
        
        // Prepare output buffers based on model metadata
        for (i in 0 until interpreter.outputTensorCount) {
            val shape = interpreter.getOutputTensor(i).shape()
            outputs[i] = Array(1) { FloatArray(shape[shape.size - 1]) }
        }

        interpreter.runForMultipleInputsOutputs(arrayOf(buffer), outputs)

        // Convert back to simple Map<Int, FloatArray> for easier consumption
        return outputs.mapValues { (_, value) ->
            (value as Array<FloatArray>)[0]
        }
    }

    override fun close() {
        interpreter.close()
    }
}
