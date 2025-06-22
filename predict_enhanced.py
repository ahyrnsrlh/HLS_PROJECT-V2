"""
Enhanced SDG Prediction System dengan database kata kunci yang lebih lengkap
"""

import re
import pandas as pd
import os

class EnhancedSDGPredictor:
    """Enhanced SDG predictor dengan rule-based yang lebih comprehensive"""
    
    def __init__(self):
        # Extended keyword database berdasarkan data asli
        self.sdg_keywords = {
            "SDG_1": {
                "name": "Tanpa Kemiskinan",
                "keywords": [
                    "poverty", "kemiskinan", "miskin", "financial inclusion", "keuangan inklusif",
                    "income", "pendapatan", "ekonomi keluarga", "kesejahteraan", "bantuan sosial",
                    "social protection", "perlindungan sosial", "employment", "pekerjaan",
                    "livelihood", "penghidupan", "microfinance", "mikro keuangan"
                ]
            },
            "SDG_2": {
                "name": "Tanpa Kelaparan", 
                "keywords": [
                    "hunger", "kelaparan", "food security", "ketahanan pangan", "nutrition", "nutrisi",
                    "malnutrition", "malnutrisi", "stunting", "gizi", "pangan", "pertanian",
                    "agriculture", "farming", "crops", "tanaman", "padi", "rice", "upland rice",
                    "soil", "tanah", "kedelai", "soy", "protein", "antioxidant", "antioksidan",
                    "food", "makanan", "diet", "dietary", "vitamin", "mineral"
                ]
            },
            "SDG_3": {
                "name": "Kehidupan Sehat dan Sejahtera",
                "keywords": [
                    "health", "kesehatan", "sehat", "medical", "medis", "disease", "penyakit",
                    "anemia", "iron", "zat besi", "deficiency", "defisiensi", "covid", "pandemic",
                    "pandemi", "therapy", "terapi", "treatment", "pengobatan", "medicine", "obat",
                    "hospital", "rumah sakit", "patient", "pasien", "clinical", "klinik",
                    "maternal", "ibu hamil", "pregnancy", "kehamilan", "persalinan", "delivery",
                    "mortality", "kematian", "epidemiological", "epidemiologi", "drug", "pharmaceutical",
                    "antibiotics", "antibiotic", "solubility", "dissolution", "mental health"
                ]
            },
            "SDG_4": {
                "name": "Pendidikan Berkualitas",
                "keywords": [
                    "education", "pendidikan", "didik", "learning", "pembelajaran", "ajar",
                    "school", "sekolah", "student", "siswa", "teacher", "guru", "curriculum",
                    "kurikulum", "e-learning", "online learning", "daring", "digital learning",
                    "module", "modul", "teaching", "pengajaran", "knowledge", "pengetahuan",
                    "skill", "keterampilan", "literacy", "literasi", "university", "universitas",
                    "academic", "akademik", "research", "penelitian", "critical thinking",
                    "berpikir kritis", "character", "karakter", "civic", "kewarganegaraan",
                    "blended learning", "inquiry", "metacognitive", "metakognitif", "tpck"
                ]
            },
            "SDG_5": {
                "name": "Kesetaraan Gender",
                "keywords": [
                    "gender", "women", "perempuan", "wanita", "female", "equality", "kesetaraan",
                    "discrimination", "diskriminasi", "empowerment", "pemberdayaan",
                    "maternal", "ibu", "mother", "leadership", "kepemimpinan wanita",
                    "workplace gender", "gender workplace", "violence against women"
                ]
            },
            "SDG_6": {
                "name": "Air Bersih dan Sanitasi",
                "keywords": [
                    "water", "air", "clean water", "air bersih", "sanitation", "sanitasi",
                    "drinking water", "air minum", "water quality", "kualitas air",
                    "water access", "akses air", "hygiene", "kebersihan", "wastewater",
                    "air limbah", "water treatment", "pengolahan air"
                ]
            },
            "SDG_7": {
                "name": "Energi Bersih dan Terjangkau",
                "keywords": [
                    "energy", "energi", "renewable", "terbarukan", "solar", "surya",
                    "wind", "angin", "power", "listrik", "electricity", "clean energy",
                    "energi bersih", "sustainable energy", "energi berkelanjutan",
                    "fossil fuel", "bahan bakar fosil", "carbon", "karbon", "emission",
                    "emisi", "grid", "jaringan listrik"
                ]
            },
            "SDG_8": {
                "name": "Pekerjaan Layak dan Pertumbuhan Ekonomi",
                "keywords": [
                    "employment", "pekerjaan", "job", "kerja", "work", "economic growth",
                    "pertumbuhan ekonomi", "ekonomi", "business", "bisnis", "entrepreneurship",
                    "kewirausahaan", "umkm", "sme", "startup", "productivity", "produktivitas",
                    "income", "pendapatan", "wage", "upah", "labor", "tenaga kerja",
                    "workforce", "angkatan kerja", "career", "karir", "performance",
                    "kinerja", "motivation", "motivasi", "stress", "brand equity",
                    "marketing", "pemasaran", "commerce", "perdagangan", "financial",
                    "keuangan", "revenue", "pendapatan"
                ]
            },
            "SDG_9": {
                "name": "Industri, Inovasi, dan Infrastruktur",
                "keywords": [
                    "industry", "industri", "innovation", "inovasi", "infrastructure", "infrastruktur",
                    "technology", "teknologi", "digital", "internet", "ict", "information technology",
                    "sistem informasi", "research development", "rd", "r&d", "manufacturing",
                    "manufaktur", "production", "produksi", "automation", "otomasi",
                    "artificial intelligence", "ai", "machine learning", "blockchain",
                    "smart city", "kota pintar", "transportation", "transportasi",
                    "communication", "komunikasi", "startup", "biosynthetic", "computational",
                    "framework", "birokrasi", "revolusi industri", "smart asn"
                ]
            },
            "SDG_10": {
                "name": "Berkurangnya Kesenjangan",
                "keywords": [
                    "inequality", "kesenjangan", "disparity", "disparitas", "inclusion",
                    "inklusi", "social inclusion", "inklusi sosial", "marginalized",
                    "terpinggirkan", "vulnerable", "rentan", "discrimination", "diskriminasi",
                    "equity", "keadilan", "fair", "adil", "income distribution"
                ]
            },
            "SDG_11": {
                "name": "Kota dan Permukiman yang Berkelanjutan",
                "keywords": [
                    "city", "kota", "urban", "perkotaan", "sustainable city", "kota berkelanjutan",
                    "housing", "perumahan", "settlement", "permukiman", "transportation",
                    "transportasi", "planning", "perencanaan kota", "infrastructure",
                    "infrastruktur kota", "slum", "kumuh", "public space", "ruang publik",
                    "disaster", "bencana", "resilience", "ketahanan", "kearifan lokal",
                    "local wisdom", "tradisi", "tradition", "heritage", "warisan"
                ]
            },
            "SDG_12": {
                "name": "Konsumsi dan Produksi yang Bertanggung Jawab",
                "keywords": [
                    "consumption", "konsumsi", "production", "produksi", "sustainable consumption",
                    "konsumsi berkelanjutan", "responsible", "bertanggung jawab", "waste",
                    "limbah", "recycling", "daur ulang", "circular economy", "ekonomi sirkular",
                    "ecolabel", "green", "hijau", "environmental", "lingkungan",
                    "purchase intention", "niat beli", "consumer behavior", "perilaku konsumen",
                    "supply chain", "rantai pasok", "lifecycle", "siklus hidup"
                ]
            },
            "SDG_13": {
                "name": "Penanganan Perubahan Iklim",
                "keywords": [
                    "climate", "iklim", "climate change", "perubahan iklim", "global warming",
                    "pemanasan global", "carbon", "karbon", "emission", "emisi",
                    "greenhouse gas", "gas rumah kaca", "mitigation", "mitigasi",
                    "adaptation", "adaptasi", "renewable energy", "energi terbarukan",
                    "fossil fuel", "bahan bakar fosil", "sustainability", "keberlanjutan",
                    "environmental impact", "dampak lingkungan"
                ]
            },
            "SDG_14": {
                "name": "Ekosistem Laut",
                "keywords": [
                    "marine", "laut", "ocean", "samudra", "sea", "laut", "aquatic", "akuatik",
                    "fish", "ikan", "fishing", "perikanan", "coral", "terumbu karang",
                    "coastal", "pesisir", "maritime", "maritim", "blue economy",
                    "ekonomi biru", "marine pollution", "pencemaran laut", "overfishing",
                    "penangkapan berlebihan", "wisata bahari", "marine tourism"
                ]
            },
            "SDG_15": {
                "name": "Ekosistem Daratan",
                "keywords": [
                    "forest", "hutan", "deforestation", "deforestasi", "biodiversity",
                    "keanekaragaman hayati", "ecosystem", "ekosistem", "wildlife", "satwa liar",
                    "conservation", "konservasi", "land", "lahan", "soil", "tanah",
                    "degradation", "degradasi", "sustainable land", "lahan berkelanjutan",
                    "terrestrial", "daratan", "flora", "fauna", "species", "spesies",
                    "gene", "gen", "biosynthetic", "repository", "cluster"
                ]
            },
            "SDG_16": {
                "name": "Perdamaian, Keadilan, dan Kelembagaan yang Tangguh",
                "keywords": [
                    "peace", "perdamaian", "justice", "keadilan", "law", "hukum",
                    "legal", "hukum", "court", "pengadilan", "governance", "tata kelola",
                    "institution", "kelembagaan", "democracy", "demokrasi", "human rights",
                    "hak asasi manusia", "corruption", "korupsi", "transparency", "transparansi",
                    "accountability", "akuntabilitas", "rule of law", "supremasi hukum",
                    "conflict", "konflik", "violence", "kekerasan", "civic", "kewarganegaraan",
                    "pancasila", "constitutional", "konstitusi", "sengketa", "dispute",
                    "communication", "komunikasi keluarga", "family communication"
                ]
            },
            "SDG_17": {
                "name": "Kemitraan untuk Mencapai Tujuan",
                "keywords": [
                    "partnership", "kemitraan", "cooperation", "kerjasama", "collaboration",
                    "kolaborasi", "international", "internasional", "global", "multilateral",
                    "development aid", "bantuan pembangunan", "capacity building",
                    "pembangunan kapasitas", "technology transfer", "transfer teknologi",
                    "south-south", "triangular cooperation", "finance", "pembiayaan",
                    "investment", "investasi", "trade", "perdagangan"
                ]
            }
        }
    
    def preprocess_text(self, text):
        """Preprocessing text untuk analisis"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        return text
    
    def predict_sdg(self, title, threshold=0.1):
        """Predict SDG dengan scoring yang lebih sensitif"""
        processed_title = self.preprocess_text(title)
        
        if not processed_title:
            return []
        
        sdg_scores = {}
        
        # Hitung score untuk setiap SDG
        for sdg_id, sdg_data in self.sdg_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in sdg_data["keywords"]:
                keyword_lower = keyword.lower()
                
                # Exact match gets highest score
                if keyword_lower in processed_title:
                    if len(keyword_lower.split()) > 1:  # Multi-word phrases get higher score
                        score += 3
                    else:
                        score += 2
                    matched_keywords.append(keyword)
                
                # Partial match for compound words
                elif any(word in processed_title for word in keyword_lower.split()):
                    score += 1
            
            # Normalize score berdasarkan jumlah kata dalam title
            title_length = len(processed_title.split())
            normalized_score = score / max(title_length, 1)
            
            if normalized_score > threshold:
                sdg_scores[sdg_id] = {
                    "score": normalized_score,
                    "name": sdg_data["name"],
                    "matched_keywords": matched_keywords
                }
        
        # Sort berdasarkan score
        sorted_sdgs = sorted(sdg_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Format output
        predictions = []
        for sdg_id, data in sorted_sdgs[:5]:  # Top 5
            predictions.append({
                "sdg": sdg_id,
                "name": data["name"],
                "confidence": min(data["score"], 1.0),  # Cap at 1.0
                "matched_keywords": data["matched_keywords"]
            })
        
        return predictions
    
    def interactive_demo(self):
        """Demo interaktif"""
        print("="*70)
        print("🎯 ENHANCED SDG PREDICTION SYSTEM")
        print("="*70)
        print("Masukkan judul penelitian untuk prediksi SDG")
        print("Ketik 'examples' untuk melihat contoh, 'quit' untuk keluar\n")
        
        # Test cases dari data asli
        examples = [
            "anemia defisiensi zat besi fe",
            "sistem pembelajaran daring fakultas keguruan",
            "sustainable energy renewable power",
            "climate change environmental impact", 
            "kearifan lokal masyarakat tradisi pernikahan",
            "komunikasi bisnis digital marketing",
            "clean water sanitation access rural",
            "poverty reduction microfinance inclusion",
            "machine learning artificial intelligence",
            "marine tourism wisata bahari"
        ]
        
        while True:
            try:
                user_input = input("📝 Masukkan judul: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Selesai!")
                    break
                
                if user_input.lower() == 'examples':
                    print("\n📋 CONTOH PREDIKSI:")
                    print("-" * 50)
                    for i, example in enumerate(examples, 1):
                        print(f"\n{i}. '{example}'")
                        preds = self.predict_sdg(example)
                        if preds:
                            for j, pred in enumerate(preds[:3], 1):
                                print(f"   {j}. {pred['sdg']}: {pred['name']} (score: {pred['confidence']:.2f})")
                        else:
                            print("   Tidak ada prediksi")
                    print("\n" + "-" * 50)
                    continue
                
                if not user_input:
                    print("⚠️ Masukkan judul yang valid")
                    continue
                
                print(f"\n🔍 Menganalisis: '{user_input}'")
                print("-" * 50)
                
                predictions = self.predict_sdg(user_input)
                
                if predictions:
                    print("🎯 Prediksi SDG:")
                    for i, pred in enumerate(predictions, 1):
                        confidence_level = "Tinggi" if pred['confidence'] > 0.7 else "Sedang" if pred['confidence'] > 0.4 else "Rendah"
                        print(f"  {i}. {pred['sdg']}: {pred['name']}")
                        print(f"     Confidence: {pred['confidence']:.2f} ({confidence_level})")
                        if pred['matched_keywords']:
                            print(f"     Kata kunci: {', '.join(pred['matched_keywords'][:5])}")
                else:
                    print("❌ Tidak ditemukan prediksi SDG yang kuat")
                    print("💡 Coba gunakan kata kunci seperti:")
                    print("   - Kesehatan: anemia, medis, penyakit, terapi")
                    print("   - Pendidikan: pembelajaran, siswa, kurikulum") 
                    print("   - Lingkungan: climate, sustainable, energi")
                    print("   - Ekonomi: bisnis, pekerjaan, ekonomi, umkm")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Program dihentikan!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    predictor = EnhancedSDGPredictor()
    predictor.interactive_demo()

if __name__ == "__main__":
    main()
