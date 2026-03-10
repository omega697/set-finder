package com.guywithburrito.setfinder.cv

import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * A detected quadrilateral and the set of strategies that found it.
 */
data class CandidateQuad(val quad: MatOfPoint2f, val foundBy: MutableSet<QuadFindingStrategy>)

/**
 * Functional strategy names for auditing pipeline contribution.
 */
interface QuadFindingStrategy {
    object Boundary : QuadFindingStrategy { override fun toString() = "Boundary" }
    object Chrominance : QuadFindingStrategy { override fun toString() = "Chrominance" }
    object Synthetic : QuadFindingStrategy { override fun toString() = "Synthetic" }
}

/**
 * Interface for finding candidate quadrilaterals in an image.
 */
interface QuadFinder {
    fun findCandidates(mat: Mat): List<MatOfPoint2f>
    fun findCandidatesFull(mat: Mat): List<CandidateQuad>
    fun findLikelyCards(mat: Mat): List<MatOfPoint2f>
}

/**
 * High-precision OpenCV-based QuadFinder using margin-based lightness/color validation.
 * Optimized to ignore background clutter (keyboards/desks) by verifying card stock.
 */
class OpenCVQuadFinder(private val settingsManager: com.guywithburrito.setfinder.tracking.SettingsManager? = null) : QuadFinder {

    override fun findCandidates(mat: Mat): List<MatOfPoint2f> {
        return findCandidatesFull(mat).map { it.quad }
    }

    override fun findCandidatesFull(mat: Mat): List<CandidateQuad> {
        val frameArea = (mat.width() * mat.height()).toDouble()
        val allCandidates = mutableListOf<CandidateQuad>()
        
        val lab = Mat()
        Imgproc.cvtColor(mat, lab, Imgproc.COLOR_RGB2Lab)
        val lChannel = Mat()
        Core.extractChannel(lab, lChannel, 0)
        
        val potentialSymbols = mutableListOf<MatOfPoint>()

        // 1. Boundary Pass: Using RETR_EXTERNAL to capture card edges while ignoring noise
        val blurredL = Mat()
        Imgproc.GaussianBlur(lChannel, blurredL, Size(5.0, 5.0), 0.0)
        val thresh = Mat()
        for (blockSize in listOf(51, 101, 151)) {
            Imgproc.adaptiveThreshold(blurredL, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, blockSize, 2.0)
            extractBoundaryQuads(thresh, lab, allCandidates, frameArea)
        }

        // 2. Chrominance Pass: Still using RETR_EXTERNAL for precision
        val a = Mat(); val b = Mat(); val aDist = Mat(); val bDist = Mat(); val colorMask = Mat()
        Core.extractChannel(lab, a, 1); Core.extractChannel(lab, b, 2)
        Core.absdiff(a, Scalar(128.0), aDist); Core.absdiff(b, Scalar(128.0), bDist)
        Core.add(aDist, bDist, colorMask)
        val colorThresh = Mat()
        Imgproc.threshold(colorMask, colorThresh, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
        extractBoundaryQuads(colorThresh, lab, allCandidates, frameArea)
        
        // 3. Symbol detection (needed for synthetic fallback) - RETR_LIST for internal shapes
        extractSymbols(colorThresh, lab, potentialSymbols, frameArea)
        for (blockSize in listOf(21, 51)) {
            Imgproc.adaptiveThreshold(blurredL, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, blockSize, 2.0)
            extractSymbols(thresh, lab, potentialSymbols, frameArea)
        }

        // 4. Synthetic Fallback
        val synthQuads = mutableListOf<MatOfPoint2f>()
        generateSyntheticQuads(potentialSymbols, synthQuads)
        synthQuads.forEach { q -> 
            if (q.isWhiteCard(lab)) {
                allCandidates.add(CandidateQuad(q, mutableSetOf(QuadFindingStrategy.Synthetic)))
            } else {
                q.release()
            }
        }

        // 5. Final Deduplication
        val finalCandidates = mergeAndDeduplicate(allCandidates)
        
        // Cleanup
        lab.release(); lChannel.release(); blurredL.release(); thresh.release()
        a.release(); b.release(); aDist.release(); bDist.release(); colorMask.release(); colorThresh.release()
        
        return finalCandidates
    }

    private fun extractBoundaryQuads(thresh: Mat, lab: Mat, results: MutableList<CandidateQuad>, frameArea: Double) {
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(thresh, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        
        val approx = MatOfPoint2f()
        val hullIndices = MatOfInt()
        
        for (cnt in contours) {
            val area = Imgproc.contourArea(cnt)
            if (area > frameArea / 20000.0 && area < frameArea * 0.9) {
                Imgproc.convexHull(cnt, hullIndices)
                val hullPoints = mutableListOf<Point>()
                val cntArray = cnt.toArray()
                val indicesArray = hullIndices.toArray()
                for (idx in indicesArray) { hullPoints.add(cntArray[idx]) }
                
                val hull = MatOfPoint(*hullPoints.toTypedArray())
                val hull2f = MatOfPoint2f(*hull.toArray())
                Imgproc.approxPolyDP(hull2f, approx, 0.02 * Imgproc.arcLength(hull2f, true), true)
                
                if (approx.total() == 4L && Imgproc.isContourConvex(MatOfPoint(*approx.toArray()))) {
                    val quad = MatOfPoint2f(*approx.toArray())
                    if (quad.isWhiteCard(lab)) {
                        results.add(CandidateQuad(quad, mutableSetOf(QuadFindingStrategy.Boundary)))
                    } else {
                        quad.release()
                    }
                }
                hull.release(); hull2f.release()
            }
        }
        approx.release(); hullIndices.release()
    }

    private fun extractSymbols(thresh: Mat, lab: Mat, symbols: MutableList<MatOfPoint>, frameArea: Double) {
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(thresh, contours, Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
        for (cnt in contours) {
            val area = Imgproc.contourArea(cnt)
            if (area > frameArea / 200000.0 && area < frameArea / 1000.0) {
                val rect = Imgproc.boundingRect(cnt)
                if (rect.x < 1 || rect.y < 1 || rect.x + rect.width + 1 > lab.cols() || rect.y + rect.height + 1 > lab.rows()) continue
                val roi = lab.submat(rect)
                val mean = Core.mean(roi); roi.release()
                val a = mean.`val`[1] - 128.0; val b = mean.`val`[2] - 128.0
                if (Math.sqrt(a * a + b * b) > 8.0) symbols.add(cnt)
            }
        }
    }

    private fun generateSyntheticQuads(symbols: List<MatOfPoint>, candidates: MutableList<MatOfPoint2f>) {
        if (symbols.isEmpty()) return
        val used = BooleanArray(symbols.size)
        for (i in symbols.indices) {
            if (used[i]) continue
            val cluster = mutableListOf<MatOfPoint>()
            cluster.add(symbols[i]); used[i] = true
            val rectI = Imgproc.boundingRect(symbols[i])
            val centerI = Point(rectI.x + rectI.width / 2.0, rectI.y + rectI.height / 2.0)
            val maxDist = Math.max(rectI.width, rectI.height) * 6.0
            for (j in i + 1 until symbols.size) {
                if (used[j]) continue
                val rectJ = Imgproc.boundingRect(symbols[j])
                val centerJ = Point(rectJ.x + rectJ.width / 2.0, rectJ.y + rectJ.height / 2.0)
                val dist = Math.sqrt(Math.pow(centerI.x - centerJ.x, 2.0) + Math.pow(centerI.y - centerJ.y, 2.0))
                if (dist < maxDist) { cluster.add(symbols[j]); used[j] = true }
            }
            if (cluster.size in 1..3) {
                val allPoints = mutableListOf<Point>()
                cluster.forEach { allPoints.addAll(it.toList()) }
                val clusterMat = MatOfPoint2f(*allPoints.toTypedArray())
                val minRect = Imgproc.minAreaRect(clusterMat); clusterMat.release()
                val expanded = minRect.clone()
                expanded.size.width *= 2.6; expanded.size.height *= 2.6
                val pts = arrayOf(Point(), Point(), Point(), Point())
                expanded.points(pts)
                candidates.add(MatOfPoint2f(*pts))
            }
        }
    }

    private fun mergeAndDeduplicate(candidates: List<CandidateQuad>): List<CandidateQuad> {
        if (candidates.isEmpty()) return emptyList()
        val sorted = candidates.sortedByDescending { Imgproc.contourArea(it.quad) }
        val unique = mutableListOf<CandidateQuad>()
        for (cand in sorted) {
            var isDuplicate = false
            for (u in unique) {
                if (cand.quad.calculateIoU(u.quad) > 0.7) {
                    u.foundBy.addAll(cand.foundBy)
                    isDuplicate = true; break
                }
            }
            if (!isDuplicate) unique.add(cand)
            if (unique.size >= 25) break 
        }
        return unique
    }

    override fun findLikelyCards(mat: Mat): List<MatOfPoint2f> {
        return findCandidates(mat)
    }
}
