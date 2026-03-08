package com.guywithburrito.setfinder.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.guywithburrito.setfinder.card.SetCard
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

class TFLiteCardIdentifier(context: Context) {
    private var filterInterpreter: Interpreter? = null
    private var expertInterpreter: Interpreter? = null
    
    // Correct v12 Mapping based on RED DIAMOND ONE SOLID prediction:
    // Output 0: Color (idx 1=RED)
    // Output 1: Shape (idx 2=DIAMOND)
    // Output 2: Count (idx 1=ONE)
    // Output 3: Pattern (idx 1=SOLID)
    private var colorIdx = 0
    private var shapeIdx = 1
    private var countIdx = 2
    private var patternIdx = 3

    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(127.5f, 127.5f))
        .build()

    init {
        try {
            filterInterpreter = Interpreter(FileUtil.loadMappedFile(context, "card_filter.tflite"))
            expertInterpreter = Interpreter(FileUtil.loadMappedFile(context, "set_card_model_final.tflite")).also {
                // Try dynamic mapping for v13 robustness
                for (i in 0 until it.outputTensorCount) {
                    val name = it.getOutputTensor(i).name()
                    Log.d("TFLite", "Output $i name: $name")
                    when {
                        name.contains("color_out") -> colorIdx = i
                        name.contains("shape_out") -> shapeIdx = i
                        name.contains("count_out") -> countIdx = i
                        name.contains("pattern_out") -> patternIdx = i
                    }
                }
                Log.d("TFLite", "Final Indices: col=$colorIdx, shp=$shapeIdx, cnt=$countIdx, pat=$patternIdx")
            }
        } catch (e: Exception) {
            Log.e("TFLite", "Init failed", e)
        }
    }

    private fun applyWhiteBalance(bitmap: Bitmap): Bitmap {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat) 
        val rgb = Mat(); Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_RGBA2RGB)
        val lab = Mat(); Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
        val channels = mutableListOf<Mat>(); Core.split(lab, channels)
        val aMean = Core.mean(channels[1]).`val`[0]
        val bMean = Core.mean(channels[2]).`val`[0]
        Core.add(channels[1], org.opencv.core.Scalar(128.0 - aMean), channels[1])
        Core.add(channels[2], org.opencv.core.Scalar(128.0 - bMean), channels[2])
        Core.merge(channels, lab)
        Imgproc.cvtColor(lab, rgb, Imgproc.COLOR_Lab2RGB)
        Imgproc.cvtColor(rgb, mat, Imgproc.COLOR_RGB2RGBA)
        val result = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, result)
        mat.release(); rgb.release(); lab.release(); channels.forEach { it.release() }
        return result
    }

    fun identifyCard(bitmap: Bitmap): SetCard? {
        val filter = filterInterpreter ?: return null
        val expert = expertInterpreter ?: return null

        try {
            val balanced = applyWhiteBalance(bitmap)
            var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
            tensorImage.load(balanced)
            tensorImage = imageProcessor.process(tensorImage)
            val buffer = tensorImage.buffer
            
            val filterOutput = Array(1) { FloatArray(1) }
            filter.run(buffer, filterOutput)
            if (filterOutput[0][0] < 0.1f) return null

            buffer.rewind()
            val outputs = mutableMapOf<Int, Any>()
            outputs[0] = Array(1) { FloatArray(4) }
            outputs[1] = Array(1) { FloatArray(4) }
            outputs[2] = Array(1) { FloatArray(4) }
            outputs[3] = Array(1) { FloatArray(4) }

            expert.runForMultipleInputsOutputs(arrayOf(buffer), outputs)

            val colorScores = (outputs[colorIdx] as Array<FloatArray>)[0]
            val shapeScores = (outputs[shapeIdx] as Array<FloatArray>)[0]
            val countScores = (outputs[countIdx] as Array<FloatArray>)[0]
            val patternScores = (outputs[patternIdx] as Array<FloatArray>)[0]

            val colIdx = colorScores.argmax(); val sIdx = shapeScores.argmax()
            val cIdx = countScores.argmax(); val pIdx = patternScores.argmax()

            if (cIdx == 0 || colIdx == 0 || pIdx == 0 || sIdx == 0) return null
            if (countScores[cIdx] < 0.3f || colorScores[colIdx] < 0.3f || 
                patternScores[pIdx] < 0.3f || shapeScores[sIdx] < 0.3f) return null

            val count = when (cIdx) { 1 -> SetCard.Count.ONE; 2 -> SetCard.Count.TWO; 3 -> SetCard.Count.THREE; else -> null } ?: return null
            val color = when (colIdx) { 1 -> SetCard.Color.RED; 2 -> SetCard.Color.GREEN; 3 -> SetCard.Color.PURPLE; else -> null } ?: return null
            val pattern = when (pIdx) { 1 -> SetCard.Pattern.SOLID; 2 -> SetCard.Pattern.SHADED; 3 -> SetCard.Pattern.EMPTY; else -> null } ?: return null
            val shape = when (sIdx) { 1 -> SetCard.Shape.OVAL; 2 -> SetCard.Shape.DIAMOND; 3 -> SetCard.Shape.SQUIGGLE; else -> null } ?: return null

            return SetCard(shape, pattern, count, color)
        } catch (e: Exception) {
            Log.e("TFLite", "Identification failed", e)
            return null
        }
    }

    private fun FloatArray.argmax(): Int {
        var bestIdx = 0; var maxVal = -1f
        for (i in indices) { if (this[i] > maxVal) { maxVal = this[i]; bestIdx = i } }
        return bestIdx
    }

    fun close() { filterInterpreter?.close(); expertInterpreter?.close() }
}
