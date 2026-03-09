package com.guywithburrito.setfinder.ml

import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import java.lang.StringBuilder

@RunWith(AndroidJUnit4::class)
class CardFilterModelInstrumentationTest {

    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(127.5f, 127.5f))
        .build()

    @Test
    fun getConfidence_isAccurateAcrossAllAssets() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val testContext = InstrumentationRegistry.getInstrumentation().context
        
        val filter = TFLiteCardFilterModel(appContext)
        val assets = testContext.assets.list("") ?: emptyArray()
        
        val errors = StringBuilder()
        var passed = 0
        var tested = 0
        
        assets.forEach { assetName ->
            if (!assetName.endsWith(".jpg")) return@forEach
            
            // Skip full scenes
            if (assetName.contains("no_set") || assetName.contains("wide_shot") || assetName.contains("sets")) return@forEach

            tested++
            val inputStream = testContext.assets.open(assetName)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            
            var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
            tensorImage.load(bitmap)
            tensorImage = imageProcessor.process(tensorImage)
            
            val confidence = filter.getConfidence(tensorImage.buffer)
            android.util.Log.d("CardFilterTest", "Asset: $assetName, confidence=$confidence")
            
            val isExpectedCard = assetName.startsWith("test_") || assetName.startsWith("card_") || assetName.startsWith("chip_")
            val isExpectedNonCard = assetName.contains("no_cards") || assetName.contains("NONE")
            
            if (isExpectedCard && confidence < 0.1f) {
                errors.append("REJECTED Card $assetName (conf=${String.format("%.3f", confidence)})\n")
            } else if (isExpectedNonCard && confidence >= 0.1f) {
                errors.append("ACCEPTED Non-Card $assetName (conf=${String.format("%.3f", confidence)})\n")
            } else {
                passed++
            }
        }
        
        filter.close()
        
        android.util.Log.d("CardFilterTest", "Passed $passed / $tested")
        if (errors.isNotEmpty()) {
            throw AssertionError("Card filter accuracy test failed with ${tested - passed} errors:\n$errors")
        }
    }
}
