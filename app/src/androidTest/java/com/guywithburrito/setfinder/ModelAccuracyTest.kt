package com.guywithburrito.setfinder

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.guywithburrito.setfinder.card.SetCard
import com.guywithburrito.setfinder.cv.OpenCVWhiteBalancer
import com.guywithburrito.setfinder.ml.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.lang.StringBuilder

@RunWith(AndroidJUnit4::class)
class ModelAccuracyTest {

    private lateinit var identifier: CardIdentifier

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        
        // Inject modular components for canonical v12 pipeline
        identifier = TFLiteCardIdentifier(
            TFLiteCardFilterModel(context),
            TFLiteExpertModel(context),
            CardModelMapper.V12,
            OpenCVWhiteBalancer()
        )
    }

    @Test
    fun testIdentificationAccuracyAcrossAllAssets() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val assets = context.assets.list("") ?: emptyArray()
        
        val errors = StringBuilder()
        var cardsTested = 0
        var cardsPassed = 0
        var nonCardsTested = 0
        var nonCardsPassed = 0
        
        assets.forEach { assetName ->
            if (!assetName.endsWith(".jpg")) return@forEach
            
            val bitmap = loadAssetAsRGB(assetName)
            val result = identifier.identifyCard(bitmap)
            
            when {
                // Case 1: Canonical test chips (test_ prefixed)
                assetName.startsWith("test_") && !assetName.contains("NONE") -> {
                    cardsTested++
                    val expected = parseLabelFromTestFile(assetName)
                    if (result == null) {
                        errors.append("FAILED $assetName (Known Card): result was null\n")
                    } else if (result != expected) {
                        errors.append("FAILED $assetName (Known Card): expected: $expected, but was: $result\n")
                    } else {
                        cardsPassed++
                    }
                }
                
                // Case 2: Other extracted chips (chip_ or card_ prefixed)
                assetName.startsWith("chip_") || assetName.startsWith("card_") -> {
                    // Skip full scene images
                    if (assetName.contains("no_set") || assetName.contains("wide_shot") || assetName.contains("sets")) return@forEach
                    
                    cardsTested++
                    val expected = parseLabelFromOtherFile(assetName)
                    if (result == null) {
                        errors.append("FAILED $assetName (Known Chip): result was null\n")
                    } else if (result != expected) {
                        errors.append("FAILED $assetName (Known Chip): expected: $expected, but was: $result\n")
                    } else {
                        cardsPassed++
                    }
                }
                
                // Case 3: Known Non-Card (desk_ or NONE prefixed)
                assetName.contains("no_cards") || assetName.contains("NONE") -> {
                    nonCardsTested++
                    if (result != null) {
                        errors.append("FAILED $assetName (Non-Card): expected null, but was: $result (False positive)\n")
                    } else {
                        nonCardsPassed++
                    }
                }
            }
        }
        
        val totalPassed = cardsPassed + nonCardsPassed
        val totalTested = cardsTested + nonCardsTested
        
        android.util.Log.d("AccuracyTest", "Passed $totalPassed / $totalTested")
        android.util.Log.d("AccuracyTest", "Cards: $cardsPassed/$cardsTested, Non-Cards: $nonCardsPassed/$nonCardsTested")
        
        if (errors.isNotEmpty()) {
            throw AssertionError("Accuracy test failed ($totalPassed/$totalTested passed):\n$errors")
        }
        
        identifier.close()
    }

    private fun loadAssetAsRGB(assetName: String): Bitmap {
        val context = InstrumentationRegistry.getInstrumentation().context
        val inputStream = context.assets.open(assetName)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        
        val matRGBA = Mat()
        Utils.bitmapToMat(bitmap, matRGBA)
        val matRGB = Mat()
        Imgproc.cvtColor(matRGBA, matRGB, Imgproc.COLOR_RGBA2RGB)
        
        val finalBmp = Bitmap.createBitmap(matRGB.cols(), matRGB.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(matRGB, finalBmp)
        
        matRGBA.release(); matRGB.release()
        return finalBmp
    }

    private fun parseLabelFromTestFile(filename: String): SetCard {
        // test_ONE_GREEN_EMPTY_DIAMOND.jpg
        val parts = filename.removePrefix("test_").removeSuffix(".jpg").split("_")
        val count = SetCard.Count.valueOf(parts[0])
        val color = SetCard.Color.valueOf(parts[1])
        val pattern = SetCard.Pattern.valueOf(parts[2])
        val shape = SetCard.Shape.valueOf(parts[3])
        return SetCard(shape, pattern, count, color)
    }

    private fun parseLabelFromOtherFile(filename: String): SetCard {
        // card_1_green_shaded_diamond.jpg or chip_1_purple_empty_oval.jpg
        val parts = filename.removeSuffix(".jpg").split("_")
        val countStr = parts[1]
        val count = when(countStr) {
            "1" -> SetCard.Count.ONE
            "2" -> SetCard.Count.TWO
            "3" -> SetCard.Count.THREE
            else -> SetCard.Count.ONE
        }
        val color = SetCard.Color.valueOf(parts[2].uppercase())
        val patternStr = parts[3].lowercase()
        val pattern = when(patternStr) {
            "striped" -> SetCard.Pattern.SHADED
            "open" -> SetCard.Pattern.EMPTY
            "solid" -> SetCard.Pattern.SOLID
            "empty" -> SetCard.Pattern.EMPTY
            "shaded" -> SetCard.Pattern.SHADED
            else -> SetCard.Pattern.SOLID
        }
        val shape = SetCard.Shape.valueOf(parts[4].uppercase())
        return SetCard(shape, pattern, count, color)
    }
}
