import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:convert';
import 'dart:math' as math;

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.green, useMaterial3: true),
      home: const PredictionScreen(),
    );
  }
}

class PredictionScreen extends StatefulWidget {
  const PredictionScreen({super.key});
  @override
  State<PredictionScreen> createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  // Ordem rigorosa do Colab: Idade, Sexo, Etnia, Auxílio, IRA, Ingresso, Escola
  final List<int> _values = [0, 0, 0, 0, 0, 0, 0];

  final List<String> _labels = [
    'Idade (18-25 anos)?',
    'Sexo Masculino?',
    'Etnia Branca?',
    'Recebe Auxílio Estudantil?',
    'I.R.A. >= 6?',
    'Ingresso por Ampla Concorrência?',
    'Origem de Escola Privada?'
  ];

  String _result = 'Aguardando Classificação';
  Color _color = Colors.grey;

  Future<void> _predict() async {
    try {
      // 1. Carregar o JSON
      final String jsonStr = await rootBundle.loadString('assets/gaussian_naive_bayes_model.json');
      final Map<String, dynamic> model = json.decode(jsonStr);

      // 2. Extrair parâmetros (Garantindo conversão para Double)
      final List<dynamic> priors = model['class_prior_'];
      final List<dynamic> thetas = model['theta_'];
      final List<dynamic> vars = model['var_'];
      final double epsilon = (model['epsilon_'] as num).toDouble();
      final List<dynamic> classes = model['classes_'];

      int nClasses = classes.length;
      int nFeatures = _values.length;

      List<double> jointLogLikelihood = [];

      for (int i = 0; i < nClasses; i++) {
        // O Scikit-Learn começa com o log da probabilidade a priori
        double logPrior = math.log(priors[i]);

        double sumLogLikelihood = 0.0;
        for (int j = 0; j < nFeatures; j++) {
          double x = _values[j].toDouble();
          double mean = (thetas[i][j] as num).toDouble();
          // Importante: A variância usada pelo sklearn é var + epsilon
          double variance = (vars[i][j] as num).toDouble() + epsilon;

          // Fórmula exata do Scikit-Learn para GaussianNB
          // -0.5 * log(2 * pi * variance) - 0.5 * ((x - mean)^2 / variance)
          double term1 = -0.5 * math.log(2 * math.pi * variance);
          double term2 = -0.5 * (math.pow(x - mean, 2) / variance);

          sumLogLikelihood += (term1 + term2);
        }

        jointLogLikelihood.add(logPrior + sumLogLikelihood);
      }

      setState(() {
        // Encontrar o índice com o MAIOR valor (ex: -7.6 > -7.8)
        int winnerIdx = 0;
        double maxVal = jointLogLikelihood[0];

        for (int i = 1; i < jointLogLikelihood.length; i++) {
          if (jointLogLikelihood[i] > maxVal) {
            maxVal = jointLogLikelihood[i];
            winnerIdx = i;
          }
        }

        // Mapear para a classe real (0 ou 1)
        int finalClass = classes[winnerIdx];

        if (finalClass == 1) {
          _result = "RISCO: EVASÃO";
          _color = Colors.red;
        } else {
          _result = "SUCESSO: NÃO EVADIDO";
          _color = Colors.green;
        }
      });

      // Logs de Auditoria para comparar com o Colab
      print("--- AUDITORIA DE PREDIÇÃO ---");
      for (int i = 0; i < nClasses; i++) {
        print("Classe ${classes[i]} (Índice $i): ${jointLogLikelihood[i]}");
      }
      var winnerIdx;
      var finalClass;
      print("Vencedor: Índice $winnerIdx (Classe $finalClass)");

    } catch (e) {
      print("ERRO CRÍTICO: $e");
      setState(() => _result = "Erro no Modelo JSON");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Preditor de Evasão [NoDropOut - Software]',
          style: TextStyle(
            fontSize: 20.0,
            fontWeight: FontWeight.bold,
            color: Colors.white,
            letterSpacing: 1.2,
          ),
        ),
        centerTitle: true,
        backgroundColor: Colors.blueGrey,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            for (int i = 0; i < _labels.length; i++)
              SwitchListTile(
                title: Text(_labels[i]),
                value: _values[i] == 1,
                onChanged: (v) => setState(() => _values[i] = v ? 1 : 0),
              ),
            const SizedBox(height: 30),
            ElevatedButton(
              onPressed: _predict,
              style: ElevatedButton.styleFrom(
                minimumSize: const Size(double.infinity, 60),
                backgroundColor: Colors.blueGrey[800],
              ),
              child: const Text('EXECUTAR PREDIÇÃO', style: TextStyle(color: Colors.white, fontSize: 18)),
            ),
            const SizedBox(height: 30),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(30),
              decoration: BoxDecoration(
                color: _color.withOpacity(0.1),
                border: Border.all(color: _color, width: 4),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                _result,
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: _color),
              ),
            ),
          ],
        ),
      ),
    );
  }
}



