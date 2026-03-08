package com.guywithburrito.setfinder

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.guywithburrito.setfinder.cv.CardFinder
import com.guywithburrito.setfinder.cv.CardUnwarper
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicInteger

@RunWith(AndroidJUnit4::class)
class ChipExtractor {

    private val finder = CardFinder()
    private val unwarper = CardUnwarper()
    private val chipCount = AtomicInteger(0)

    @Test
    fun extractChipsFromVideos() {
        OpenCVLoader.initDebug()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        
        // Use internal cache directory to avoid permission issues
        val inputDir = context.externalCacheDir ?: return
        val outputDir = File(inputDir, "chips")
        if (!outputDir.exists()) outputDir.mkdirs()

        Log.i("ChipExtractor", "Searching for videos in ${inputDir.absolutePath}")
        val videos = inputDir.listFiles { _, name -> name.endsWith(".mp4") || name.endsWith(".mov") }
        
        if (videos.isNullOrEmpty()) {
            Log.e("ChipExtractor", "No videos found in ${inputDir.absolutePath}")
            return
        }

        for (videoFile in videos) {
            Log.i("ChipExtractor", "Processing video: ${videoFile.name}")
            processVideo(videoFile, outputDir)
        }
        
        Log.i("ChipExtractor", "Extraction complete. Total chips: ${chipCount.get()}")
    }

    private fun processVideo(videoFile: File, outputDir: File) {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(videoFile.absolutePath)
            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val durationMs = durationStr?.toLong() ?: 0L
            
            // Process a frame every 300ms for more data
            val intervalMs = 300L
            for (timeMs in 0 until durationMs step intervalMs) {
                val frameBitmap = retriever.getFrameAtTime(timeMs * 1000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                if (frameBitmap != null) {
                    processFrame(frameBitmap, outputDir)
                }
            }
        } catch (e: Exception) {
            Log.e("ChipExtractor", "Error processing ${videoFile.name}", e)
        } finally {
            retriever.release()
        }
    }

    private fun processFrame(bitmap: Bitmap, outputDir: File) {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        
        val maxDim = 1000.0
        val scale = maxDim / Math.max(mat.width(), mat.height())
        val analysisFrame = Mat()
        Imgproc.resize(mat, analysisFrame, Size(), scale, scale, Imgproc.INTER_AREA)

        val candidates = finder.findCandidates(analysisFrame)
        
        for (quad in candidates) {
            val scaledPoints = quad.toArray().map { p -> org.opencv.core.Point(p.x / scale, p.y / scale) }
            val highResQuad = org.opencv.core.MatOfPoint2f(*scaledPoints.toTypedArray())
            
            val warped = unwarper.unwarp(mat, highResQuad)
            
            val chip = Mat()
            Imgproc.resize(warped, chip, Size(144.0, 224.0))
            
            saveMatAsJpeg(chip, outputDir, "chip_${chipCount.getAndIncrement()}.jpg")
        }
    }

    private fun saveMatAsJpeg(mat: Mat, dir: File, name: String) {
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bitmap)
        val file = File(dir, name)
        FileOutputStream(file).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
        }
    }
}
