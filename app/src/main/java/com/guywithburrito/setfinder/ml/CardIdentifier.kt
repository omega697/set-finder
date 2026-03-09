package com.guywithburrito.setfinder.ml

import android.graphics.Bitmap
import com.guywithburrito.setfinder.card.SetCard
import com.guywithburrito.setfinder.cv.WhiteBalancer
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp

/**
 * High-level interface for identifying a card from a pre-extracted chip.
 */
interface CardIdentifier {
    fun identifyCard(chip: Bitmap): SetCard?
    fun close()
}

/**
 * TFLite-based implementation that orchestrates preprocessing, 
 * expert model execution, and attribute mapping.
 */
class TFLiteCardIdentifier(
    private val filterModel: CardFilterModel,
    private val expertModel: CardExpertModel,
    private val mapper: CardModelMapper,
    private val whiteBalancer: WhiteBalancer
) : CardIdentifier {

    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(127.5f, 127.5f))
        .build()

    override fun identifyCard(chip: Bitmap): SetCard? {
        // 1. Preprocess for TFLite
        var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
        tensorImage.load(chip)
        tensorImage = imageProcessor.process(tensorImage)
        val buffer = tensorImage.buffer

        // 2. Filter (Is it a card?)
        val confidence = filterModel.getConfidence(buffer)
        if (confidence < 0.1f) return null

        // 3. White Balance (Native Lab-based) - ONLY for actual cards
        val mat = Mat()
        Utils.bitmapToMat(chip, mat)
        val balancedMat = whiteBalancer.balanceRGB(mat)
        val balancedBmp = Bitmap.createBitmap(balancedMat.cols(), balancedMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(balancedMat, balancedBmp)
        
        // 4. Reprocess balanced image for Expert Model
        var expertTensor = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
        expertTensor.load(balancedBmp)
        expertTensor = imageProcessor.process(expertTensor)
        
        // 5. Expert Model Inference
        val predictions = expertModel.predict(expertTensor.buffer)
        
        // 6. Mapping
        val colIdx = mapper.argmax(predictions[0]!!)
        val shpIdx = mapper.argmax(predictions[1]!!)
        val cntIdx = mapper.argmax(predictions[2]!!)
        val patIdx = mapper.argmax(predictions[3]!!)
        
        // Cleanup native resources
        mat.release(); balancedMat.release()
        
        return mapper.mapIndices(colIdx, shpIdx, cntIdx, patIdx)
    }

    override fun close() {
        filterModel.close()
        expertModel.close()
    }
}
