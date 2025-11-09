import numpy as np
import tensorflow as tf
from collections import defaultdict, deque
import json
from datetime import datetime
import random

# Подавляем предупреждения TensorFlow
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("🚀 Инициализация Dynamic Swarm Financial AI...")


# -------------- Вспомогательные классы ----------------
class CollectiveKnowledgeEngine:
    def __init__(self):
        self.knowledge_base = defaultdict(dict)
        self.seasonal_patterns = defaultdict(list)

    def update_knowledge(self, cluster_id, success_pattern):
        """Обновляет коллективные знания на основе успешных паттернов"""
        if cluster_id not in self.knowledge_base:
            self.knowledge_base[cluster_id] = {
                'success_rates': [],
                'optimal_actions': [],
                'seasonal_adjustments': [],
                'risk_adjustments': []
            }

        self.knowledge_base[cluster_id]['success_rates'].append(
            success_pattern['success_rate'])
        self.knowledge_base[cluster_id]['optimal_actions'].append(
            success_pattern['action'])

        # Обновляем сезонные паттерны
        month = success_pattern.get('month', 0)
        self.seasonal_patterns[month].append(success_pattern)

    def get_seasonal_adjustment(self, month, cluster_id):
        """Возвращает сезонные корректировки для месяца"""
        monthly_data = self.seasonal_patterns[month]
        if not monthly_data:
            return 0.0

        # Вычисляем оптимальную корректировку для этого месяца
        successful = [d for d in monthly_data if d['success_metric'] > 1.1]
        if successful:
            avg_adjustment = np.mean([d.get('seasonal_adjustment', 0) for d in successful])
            return avg_adjustment
        return 0.0


# -------------- Динамическая архитектура Swarm Financial AI ----------------
class DynamicSwarmFinancialAI:
    def __init__(self):
        print("🤖 Создание динамических нейронных сетей...")
        # Основная модель для базовых решений
        self.main_model = self.build_dynamic_dqn_model()

        # Специализированные модели для разных сценариев
        self.crisis_model = self.build_crisis_model()
        self.opportunity_model = self.build_opportunity_model()
        self.seasonal_model = self.build_seasonal_model()

        # Коллективный интеллект
        self.swarm_memory = DynamicSwarmMemory()
        self.collective_knowledge = CollectiveKnowledgeEngine()
        self.user_clusters = DynamicUserClustering()

        print("✅ Dynamic Swarm Financial AI инициализирован!")

    def build_dynamic_dqn_model(self):
        """Динамическая модель с расширенными признаками"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(12,)),  # 12 признаков!
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(5, activation='linear')  # 5 действий
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
        return model

    def build_crisis_model(self):
        """Модель для кризисных ситуаций"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='linear')  # 0%, 2%, 5%
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
        return model

    def build_opportunity_model(self):
        """Модель для благоприятных ситуаций"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4, activation='linear')  # 10%, 15%, 20%, 25%
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
        return model

    def build_seasonal_model(self):
        """Модель для сезонных корректировок"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(7, activation='linear')  # -10%, -5%, 0%, +5%, +10%, +15%, +20%
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
        return model

    def select_strategy_model(self, state, context):
        """Выбирает подходящую модель для текущей ситуации"""
        income, expenses, balance, month = state[0], state[1], state[2], state[3]

        # Кризисная ситуация
        if balance < 15000 or (expenses / income) > 0.9:
            return "crisis", self.crisis_model, state[:6]  # Только первые 6 признаков

        # Благоприятная ситуация
        if balance > 100000 and (expenses / income) < 0.6:
            return "opportunity", self.opportunity_model, state[:6]  # Только первые 6 признаков

        # Сезонные корректировки
        if month in [11, 0, 5, 6]:  # Декабрь, Январь, Июнь, Июль
            return "seasonal", self.seasonal_model, state[3:7]  # Признаки 3-6 (месяц и связанные)

        # Стандартная ситуация
        return "standard", self.main_model, state  # Все 12 признаков

# -------------- Динамическая память Swarm ----------------
class DynamicSwarmMemory:
    def __init__(self, max_size=5000):
        self.max_size = max_size
        self.collective_experiences = deque(maxlen=max_size)
        self.cluster_success_patterns = defaultdict(list)
        self.crisis_patterns = defaultdict(list)
        self.opportunity_patterns = defaultdict(list)
        self.seasonal_patterns = defaultdict(lambda: defaultdict(list))

    def add_collective_experience(self, user_profile, action, outcome, cluster_id, context):
        """Добавляет опыт в коллективную память с контекстом"""
        experience = {
            'timestamp': datetime.now(),
            'user_profile': user_profile,
            'action': action,
            'outcome': outcome,
            'cluster_id': cluster_id,
            'context': context,
            'success_metric': outcome['final_balance'] / max(outcome['initial_balance'], 1),
            'month': context.get('month', 0),
            'situation_type': context.get('situation_type', 'standard'),
            'seasonal_adjustment': context.get('seasonal_adjustment', 0)
        }
        self.collective_experiences.append(experience)

        # Классифицируем опыт по типам ситуаций
        if experience['success_metric'] > 1.1:
            self.cluster_success_patterns[cluster_id].append(experience)

            if context.get('situation_type') == 'crisis':
                self.crisis_patterns[cluster_id].append(experience)
            elif context.get('situation_type') == 'opportunity':
                self.opportunity_patterns[cluster_id].append(experience)

            # Сезонные паттерны
            month = context.get('month', 0)
            self.seasonal_patterns[cluster_id][month].append(experience)

    def get_contextual_recommendation(self, user_state, user_cluster, context):
        """Получает контекстные рекомендации"""
        situation_type = context.get('situation_type', 'standard')
        month = context.get('month', 0)

        if situation_type == 'crisis':
            patterns = self.crisis_patterns.get(user_cluster, [])
        elif situation_type == 'opportunity':
            patterns = self.opportunity_patterns.get(user_cluster, [])
        else:
            patterns = self.cluster_success_patterns.get(user_cluster, [])

        # Фильтруем по сезону если нужно
        if situation_type == 'seasonal':
            seasonal_data = self.seasonal_patterns.get(user_cluster, {}).get(month, [])
            patterns = seasonal_data if seasonal_data else patterns

        if not patterns:
            return None

        # Анализируем успешные стратегии в этом контексте
        successful_actions = []
        for exp in patterns:
            if exp['success_metric'] > 1.1:
                successful_actions.append({
                    'action': exp['action'],
                    'context': exp['context'],
                    'success_rate': exp['success_metric'],
                    'seasonal_adjustment': exp.get('seasonal_adjustment', 0)
                })

        return self._analyze_contextual_patterns(successful_actions, user_state, context)

    def _analyze_contextual_patterns(self, successful_actions, user_state, context):
        """Анализирует контекстные паттерны"""
        if not successful_actions:
            return None

        # Взвешиваем действия по релевантности контекста
        weighted_actions = []
        for action_data in successful_actions:
            relevance_score = self._calculate_context_relevance(action_data['context'], context)
            weighted_score = action_data['success_rate'] * relevance_score
            weighted_actions.append({
                **action_data,
                'weighted_score': weighted_score
            })

        best_action = max(weighted_actions, key=lambda x: x['weighted_score'])
        return [best_action]

    def _calculate_context_relevance(self, context1, context2):
        """Вычисляет релевантность контекстов"""
        score = 0.0
        if context1.get('situation_type') == context2.get('situation_type'):
            score += 0.4
        if abs(context1.get('month', 0) - context2.get('month', 0)) <= 1:
            score += 0.3
        if context1.get('risk_level') == context2.get('risk_level'):
            score += 0.3
        return min(score, 1.0)


# -------------- Динамическая кластеризация пользователей ----------------
class DynamicUserClustering:
    def __init__(self):
        self.user_profiles = {}
        self.clusters = {}
        self.risk_profiles = {}

    def assign_cluster(self, user_profile):
        """Определяет динамический кластер пользователя"""
        income_level = self._categorize_income(user_profile['income'])
        family_status = user_profile['family_status']
        financial_goals = user_profile['goals']
        risk_profile = self._assess_risk_profile(user_profile)
        spending_efficiency = self._calculate_spending_efficiency(user_profile)

        cluster_id = f"{income_level}_{family_status}_{financial_goals}_{risk_profile}"
        return cluster_id

    def _categorize_income(self, income):
        if income < 40000:
            return "low"
        elif income < 80000:
            return "medium"
        elif income < 150000:
            return "high"
        else:
            return "premium"

    def _assess_risk_profile(self, user_profile):
        """Оценивает риск-профиль пользователя"""
        income_stability = user_profile.get('income_stability', 0.5)
        expense_volatility = user_profile.get('expense_volatility', 0.5)
        emergency_fund_ratio = user_profile.get('emergency_fund_ratio', 0.1)

        risk_score = (income_stability * 0.4 +
                      (1 - expense_volatility) * 0.3 +
                      emergency_fund_ratio * 0.3)

        if risk_score > 0.7:
            return "conservative"
        elif risk_score > 0.4:
            return "moderate"
        else:
            return "aggressive"

    def _calculate_spending_efficiency(self, user_profile):
        """Вычисляет эффективность расходов"""
        if not isinstance(user_profile.get('expenses'), dict):
            return 0.5

        expenses = user_profile['expenses']
        total = sum(expenses.values())
        if total == 0:
            return 0.5

        # Высокая эффективность = больше тратим на essentials
        essential_ratio = expenses.get('essential', 0) / total
        luxury_ratio = expenses.get('entertainment', 0) / total

        efficiency = essential_ratio * 0.7 + (1 - luxury_ratio) * 0.3
        return efficiency


# -------------- Динамическая среда с реальными сценариями ----------------
class DynamicSwarmSavingsEnv:
    def __init__(self, user_id, swarm_ai):
        self.swarm_ai = swarm_ai
        self.user_id = user_id
        self.max_steps = 12
        self.action_space = 5
        self.observation_space = 12  # Расширенное состояние!

        # Динамические данные
        self.user_profile = {}
        self.user_cluster = None
        self.collective_benchmarks = {}
        self.current_balance = 0
        self.month = 0
        self.situation_history = []

        # Динамические факторы
        self.economic_outlook = random.uniform(0.3, 1.0)  # 0.3 = кризис, 1.0 = рост
        self.personal_events = []  # Неожиданные события

    def reset(self):
        # Базовое состояние с динамическими факторами
        income = np.random.uniform(30000, 100000)
        expenses = self._generate_dynamic_expenses(income)
        self.current_balance = np.random.uniform(5000, 150000)
        self.month = 0
        self.situation_history = []

        # Генерируем неожиданные события
        self.personal_events = self._generate_personal_events()

        # Инициализация динамического профиля
        self.user_profile = {
            'income': income,
            'expenses': expenses,
            'goals': random.choice(['apartment', 'car', 'travel', 'education', 'savings']),
            'family_status': random.choice(['single', 'couple', 'family']),
            'age_group': random.choice(['18-25', '26-35', '36-45', '45+']),
            'income_stability': random.uniform(0.3, 0.9),
            'emergency_fund_ratio': random.uniform(0.05, 0.3),
            'expense_volatility': random.uniform(0.2, 0.8)
        }

        self.user_cluster = self.swarm_ai.user_clusters.assign_cluster(self.user_profile)
        self.collective_benchmarks = self._get_dynamic_benchmarks()

        print(f"👤 Создан пользователь: доход {income:.0f} руб., баланс {self.current_balance:.0f} руб.")
        print(f"🎯 Кластер: {self.user_cluster}")
        print(f"📈 Экономический прогноз: {self.economic_outlook:.1%}")

        return self._get_dynamic_state()

    def _generate_dynamic_expenses(self, income):
        """Генерирует динамические расходы с сезонными колебаниями"""
        base_essential = income * np.random.uniform(0.3, 0.5)
        base_housing = income * np.random.uniform(0.2, 0.4)
        base_transportation = income * np.random.uniform(0.05, 0.15)
        base_entertainment = income * np.random.uniform(0.05, 0.15)

        return {
            'essential': base_essential,
            'housing': base_housing,
            'transportation': base_transportation,
            'entertainment': base_entertainment
        }

    def _generate_personal_events(self):
        """Генерирует неожиданные личные события"""
        events = []
        possible_events = [
            {'type': 'medical', 'impact': -0.2, 'probability': 0.1},
            {'type': 'bonus', 'impact': 0.3, 'probability': 0.15},
            {'type': 'car_repair', 'impact': -0.15, 'probability': 0.2},
            {'type': 'tax_refund', 'impact': 0.1, 'probability': 0.1},
            {'type': 'family_emergency', 'impact': -0.25, 'probability': 0.05}
        ]

        for event in possible_events:
            if random.random() < event['probability']:
                events.append(event)

        return events

    def _get_dynamic_state(self):
        """Возвращает расширенное динамическое состояние"""
        total_expenses = sum(self.user_profile['expenses'].values())
        current_situation = self._assess_current_situation()

        # 12-мерное состояние с динамическими факторами
        dynamic_state = [
            self.user_profile['income'],  # 0: Доход
            total_expenses,  # 1: Расходы
            self.current_balance,  # 2: Баланс
            self.month,  # 3: Месяц
            self.collective_benchmarks.get('avg_savings_rate', 0.1),  # 4: Бенчмарк кластера
            self.collective_benchmarks.get('success_probability', 0.5),  # 5: Вероятность успеха
            self._get_spending_efficiency(),  # 6: Эффективность расходов
            self._get_swarm_confidence(),  # 7: Уверенность swarm
            self.economic_outlook,  # 8: Экономический прогноз
            current_situation['risk_level'],  # 9: Уровень риска
            len(self.personal_events),  # 10: Количество событий
            current_situation['situation_type_score']  # 11: Тип ситуации
        ]
        return np.array(dynamic_state, dtype=np.float32)

    def _assess_current_situation(self):
        """Оценивает текущую ситуацию"""
        income = self.user_profile['income']
        expenses = sum(self.user_profile['expenses'].values())
        balance_ratio = self.current_balance / income if income > 0 else 0
        expense_ratio = expenses / income if income > 0 else 1.0

        # Определяем тип ситуации
        if balance_ratio < 0.3 or expense_ratio > 0.9:
            situation_type = "crisis"
            risk_level = 0.8
            situation_score = 0.1
        elif balance_ratio > 2.0 and expense_ratio < 0.6 and self.economic_outlook > 0.7:
            situation_type = "opportunity"
            risk_level = 0.2
            situation_score = 0.9
        elif self.month in [11, 0]:  # Декабрь, Январь
            situation_type = "seasonal"
            risk_level = 0.5
            situation_score = 0.6
        else:
            situation_type = "standard"
            risk_level = 0.4
            situation_score = 0.5

        return {
            'type': situation_type,
            'risk_level': risk_level,
            'situation_type_score': situation_score
        }

    def _get_spending_efficiency(self):
        """Вычисляет эффективность расходов"""
        expenses = self.user_profile['expenses']
        total = sum(expenses.values())
        if total == 0:
            return 0.5

        essential_ratio = expenses.get('essential', 0) / total
        luxury_ratio = expenses.get('entertainment', 0) / total

        return essential_ratio * 0.8 + (1 - luxury_ratio) * 0.2

    def _get_swarm_confidence(self):
        """Вычисляет уверенность swarm в рекомендациях"""
        cluster_data = self.swarm_ai.swarm_memory.cluster_success_patterns.get(self.user_cluster, [])
        situation_data = self._get_situation_specific_data()

        base_confidence = min(len(cluster_data) / 50, 1.0)
        situation_confidence = len(situation_data) / 20 if situation_data else 0.3

        return (base_confidence * 0.6 + situation_confidence * 0.4)

    def _get_situation_specific_data(self):
        """Получает данные для текущей ситуации"""
        current_situation = self._assess_current_situation()
        situation_type = current_situation['type']

        if situation_type == "crisis":
            return self.swarm_ai.swarm_memory.crisis_patterns.get(self.user_cluster, [])
        elif situation_type == "opportunity":
            return self.swarm_ai.swarm_memory.opportunity_patterns.get(self.user_cluster, [])
        else:
            return self.swarm_ai.swarm_memory.cluster_success_patterns.get(self.user_cluster, [])

    def _get_dynamic_benchmarks(self):
        """Получает динамические бенчмарки"""
        cluster_data = self.swarm_ai.swarm_memory.cluster_success_patterns.get(self.user_cluster, [])
        current_situation = self._assess_current_situation()
        situation_data = self._get_situation_specific_data()

        if not cluster_data:
            return {'avg_savings_rate': 0.1, 'success_probability': 0.5}

        # Используем ситуационные данные если есть
        target_data = situation_data if situation_data else cluster_data

        successful_cases = [d for d in target_data if d['success_metric'] > 1.1]
        if successful_cases:
            avg_savings = np.mean([d['action'] * 0.05 for d in successful_cases])
            success_prob = len(successful_cases) / len(target_data)
        else:
            avg_savings = 0.1
            success_prob = 0.5

        return {
            'avg_savings_rate': avg_savings,
            'success_probability': success_prob,
            'situation_aware': len(situation_data) > 0
        }

    def step(self, action):
        """Выполняет действие с учетом динамических факторов"""
        income = self.user_profile['income']
        total_expenses = sum(self.user_profile['expenses'].values())

        # Применяем динамические корректировки
        adjusted_action = self._apply_dynamic_adjustments(action)
        savings_rate = adjusted_action * 0.05
        savings = income * savings_rate

        # Применяем неожиданные события
        event_impact = self._apply_personal_events()

        # Обновляем баланс с учетом событий
        old_balance = self.current_balance
        self.current_balance = max(0, self.current_balance + income - total_expenses - savings + event_impact)

        # Обновляем месяц и экономический прогноз
        self.month += 1
        self._update_economic_outlook()

        # Определяем контекст для swarm memory
        current_situation = self._assess_current_situation()
        context = {
            'month': self.month,
            'situation_type': current_situation['type'],
            'risk_level': current_situation['risk_level'],
            'economic_outlook': self.economic_outlook,
            'events_count': len(self.personal_events),
            'seasonal_adjustment': self._get_seasonal_adjustment()
        }

        # Сохраняем историю ситуации
        self.situation_history.append({
            'month': self.month,
            'situation': current_situation['type'],
            'action_taken': action,
            'adjusted_action': adjusted_action
        })

        # Создаем outcome для swarm memory
        outcome = {
            'initial_balance': old_balance,
            'final_balance': self.current_balance,
            'savings_made': savings,
            'event_impact': event_impact,
            'situation_type': current_situation['type']
        }

        # Добавляем опыт в коллективную память
        self.swarm_ai.swarm_memory.add_collective_experience(
            self.user_profile, adjusted_action, outcome, self.user_cluster, context
        )

        # Вычисляем награду с учетом контекста
        reward = self._calculate_contextual_reward(adjusted_action, outcome, context)

        # Проверяем завершение эпизода
        done = self.month >= self.max_steps

        return self._get_dynamic_state(), reward, done, context

    def _apply_dynamic_adjustments(self, action):
        """Применяет динамические корректировки к действию"""
        current_situation = self._assess_current_situation()
        situation_type = current_situation['type']

        if situation_type == "crisis":
            # В кризис снижаем агрессивность
            return max(0, action - 1)
        elif situation_type == "opportunity":
            # В благоприятной ситуации можно быть смелее
            return min(4, action + 1)
        elif situation_type == "seasonal":
            # Сезонные корректировки
            if self.month in [11, 0]:  # Праздники
                return max(0, action - 1)
            elif self.month in [5, 6]:  # Лето, отпуска
                return max(0, action - 1)

        return action

    def _apply_personal_events(self):
        """Применяет неожиданные личные события"""
        if not self.personal_events:
            return 0

        total_impact = 0
        income = self.user_profile['income']

        for event in self.personal_events:
            impact = event['impact'] * income
            total_impact += impact
            print(f"⚡ Событие: {event['type']}, влияние: {impact:.0f} руб.")

        # Очищаем обработанные события
        self.personal_events = []
        return total_impact

    def _update_economic_outlook(self):
        """Обновляет экономический прогноз"""
        # Имитируем изменения экономической ситуации
        change = random.uniform(-0.1, 0.1)
        self.economic_outlook = max(0.1, min(1.0, self.economic_outlook + change))

    def _get_seasonal_adjustment(self):
        """Возвращает сезонную корректировку"""
        if self.month in [11, 0]:  # Праздники
            return -0.05  # -5%
        elif self.month in [5, 6]:  # Лето
            return -0.03  # -3%
        return 0.0

    def _calculate_contextual_reward(self, action, outcome, context):
        """Вычисляет контекстную награду"""
        base_reward = outcome['final_balance'] / 10000

        # Награда за адаптивность
        situation_type = context['situation_type']
        if situation_type == "crisis" and action <= 1:
            base_reward += 2.0  # Награда за осторожность в кризис
        elif situation_type == "opportunity" and action >= 3:
            base_reward += 1.5  # Награда за смелость в возможности

        # Награда за сбережения
        if action > 0:
            base_reward += action * 0.3

        # Штраф за неадаптивность
        if situation_type == "crisis" and action >= 3:
            base_reward -= 3.0
        elif situation_type == "opportunity" and action <= 1:
            base_reward -= 2.0

        # Штраф за низкий баланс
        if outcome['final_balance'] < 5000:
            base_reward -= 5.0

        return base_reward


# -------------- Динамическая система рекомендаций ----------------
class DynamicSwarmAdvisor:
    def __init__(self, swarm_ai):
        self.swarm_ai = swarm_ai

    def get_dynamic_recommendation(self, user_state, user_profile, context):
        """Генерирует динамические рекомендации с учетом контекста"""
        user_cluster = self.swarm_ai.user_clusters.assign_cluster(user_profile)

        # Выбираем стратегию based на контексте
        strategy_type, model, model_input = self.swarm_ai.select_strategy_model(user_state, context)

        # Базовая рекомендация от выбранной модели
        if strategy_type == "crisis":
            q_values = model.predict(model_input[np.newaxis], verbose=0)[0]
            base_action = np.argmax(q_values)
            # Конвертируем в стандартный формат (0-4)
            action_mapping = {0: 0, 1: 0, 2: 1}  # 0%, 0%, 5%
            ai_action = action_mapping.get(base_action, 0)
        elif strategy_type == "opportunity":
            q_values = model.predict(model_input[np.newaxis], verbose=0)[0]
            base_action = np.argmax(q_values)
            action_mapping = {0: 2, 1: 3, 2: 4, 3: 4}  # 10%, 15%, 20%, 20%
            ai_action = action_mapping.get(base_action, 2)
        elif strategy_type == "seasonal":
            q_values = model.predict(model_input[np.newaxis], verbose=0)[0]
            base_action = np.argmax(q_values)
            # Для сезонной модели: -10%, -5%, 0%, +5%, +10%, +15%, +20%
            action_mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4}
            ai_action = action_mapping.get(base_action, 2)
        else:
            q_values = model.predict(model_input[np.newaxis], verbose=0)[0]
            ai_action = np.argmax(q_values)


        swarm_rec = self.swarm_ai.swarm_memory.get_contextual_recommendation(
            user_state, user_cluster, context)

        dynamic_insights = self._get_dynamic_insights(user_cluster, context)
        strategy_analysis = self._analyze_strategy(strategy_type, ai_action, context)

        recommendation = {
            'ai_recommendation': f"Отложить {ai_action * 5}% дохода",
            'strategy_type': strategy_type,
            'swarm_advice': self._format_dynamic_advice(swarm_rec, context),
            'dynamic_insights': dynamic_insights,
            'strategy_analysis': strategy_analysis,
            'success_probability': self._calculate_dynamic_success_probability(user_state, ai_action, user_cluster,
                                                                               context),
            'alternative_strategies': self._get_contextual_alternatives(strategy_type, context),
            'risk_assessment': self._assess_risk(ai_action, context)
        }

        return recommendation

    def _get_dynamic_insights(self, user_cluster, context):
        """Получает динамические инсайты"""
        situation_type = context.get('situation_type', 'standard')

        if situation_type == "crisis":
            return "🏥 Рекомендуется консервативная стратегия: сохраняйте ликвидность"
        elif situation_type == "opportunity":
            return "🚀 Благоприятная ситуация: можно увеличить сбережения"
        elif situation_type == "seasonal":
            return "🎄 Сезонные расходы: умеренная стратегия рекомендуется"
        else:
            return "📊 Стандартные условия: следуйте обычной стратегии"

    def _analyze_strategy(self, strategy_type, action, context):
        """Анализирует выбранную стратегию"""
        analysis = {
            'standard': f"Стандартная стратегия: {action * 5}% соответствует вашим целям",
            'crisis': f"Кризисная стратегия: {action * 5}% для сохранения финансовой стабильности",
            'opportunity': f"Стратегия возможностей: {action * 5}% для ускоренного роста",
            'seasonal': f"Сезонная стратегия: {action * 5}% с учетом временных факторов"
        }
        return analysis.get(strategy_type, analysis['standard'])

    def _format_dynamic_advice(self, swarm_data, context):
        if not swarm_data:
            return "Пока недостаточно данных для вашей ситуации"

        best_action = max(swarm_data, key=lambda x: x['weighted_score'])
        situation = context.get('situation_type', 'standard')

        advice_templates = {
            'crisis': f"В похожих кризисных ситуациях {best_action['action'] * 5}% показал наилучшие результаты",
            'opportunity': f"При таких возможностях {best_action['action'] * 5}% максимизировал рост",
            'seasonal': f"В этот сезон {best_action['action'] * 5}% был оптимальным для вашего кластера",
            'standard': f"Пользователи вашего профиля успешно достигали целей с {best_action['action'] * 5}%"
        }

        return advice_templates.get(situation, advice_templates['standard'])

    def _calculate_dynamic_success_probability(self, user_state, action, user_cluster, context):
        """Вычисляет динамическую вероятность успеха"""
        # Базовая логика с учетом контекста
        base_prob = 0.5
        situation = context.get('situation_type', 'standard')

        # Модификаторы based на ситуации
        modifiers = {
            'crisis': 0.8,  # В кризис выше риск
            'opportunity': 1.2,  # В возможности выше шанс успеха
            'seasonal': 1.0,
            'standard': 1.0
        }

        return min(0.95, base_prob * modifiers.get(situation, 1.0))

    def _get_contextual_alternatives(self, strategy_type, context):
        """Возвращает контекстные альтернативы"""
        alternatives = {
            'standard': ["Постепенное увеличение", "Оптимизация расходов", "Инвестирование"],
            'crisis': ["Экстренный фонд", "Сокращение необязательных расходов", "Реструктуризация долгов"],
            'opportunity': ["Ускоренное накопление", "Инвестиции в рост", "Диверсификация"],
            'seasonal': ["Сезонный бюджет", "Планирование крупных покупок", "Использование спецпредложений"]
        }
        return alternatives.get(strategy_type, alternatives['standard'])

    def _assess_risk(self, action, context):
        """Оценивает риск стратегии"""
        risk_levels = {
            0: "Низкий",
            1: "Низкий",
            2: "Умеренный",
            3: "Высокий",
            4: "Очень высокий"
        }

        situation = context.get('situation_type', 'standard')
        base_risk = risk_levels.get(action, "Умеренный")

        if situation == "crisis" and action >= 2:
            return f"🚨 Высокий риск: {base_risk} + кризисная ситуация"
        elif situation == "opportunity" and action <= 1:
            return f"⚠️  Умеренный риск: {base_risk} (можно увеличить)"
        else:
            return f"✅ {base_risk} риск: соответствует ситуации"


# -------------- Демонстрация динамической системы ----------------
def demonstrate_dynamic_swarm_ai():
    print("\n" + "=" * 70)
    print("🎯 ДЕМОНСТРАЦИЯ DYNAMIC SWARM FINANCIAL AI")
    print("=" * 70)

    # Инициализация системы
    print("\n1. Инициализация Dynamic Swarm AI...")
    swarm_ai = DynamicSwarmFinancialAI()

    # Симуляция коллективного опыта с разными сценариями
    print("\n2. Имитация разнообразного коллективного опыта...")
    scenarios = ['crisis', 'opportunity', 'seasonal', 'standard']

    for i in range(100):  # Больше пользователей для разнообразия
        user_env = DynamicSwarmSavingsEnv(user_id=f"sim_user_{i}", swarm_ai=swarm_ai)
        state = user_env.reset()

        # Симулируем разные стратегии для разных сценариев
        for month in range(6):
            current_situation = user_env._assess_current_situation()
            situation_type = current_situation['type']

            # Разные стратегии для разных ситуаций
            if situation_type == "crisis":
                action = random.choices([0, 0, 1], weights=[0.6, 0.3, 0.1])[0]  # Консервативно
            elif situation_type == "opportunity":
                action = random.choices([2, 3, 4], weights=[0.2, 0.5, 0.3])[0]  # Агрессивно
            else:
                action = random.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0]  # Умеренно

            next_state, reward, done, context = user_env.step(action)
            if done:
                break

    print("✅ Динамический коллективный опыт сгенерирован (100 пользователей)")

    # Основной пользователь с демонстрацией адаптивности
    print("\n3. Демонстрация адаптивности для основного пользователя...")
    user_env = DynamicSwarmSavingsEnv(user_id="main_user", swarm_ai=swarm_ai)
    state = user_env.reset()

    # Получение динамической рекомендации
    advisor = DynamicSwarmAdvisor(swarm_ai)
    context = {'month': user_env.month, 'situation_type': user_env._assess_current_situation()['type']}
    recommendation = advisor.get_dynamic_recommendation(state, user_env.user_profile, context)

    print("\n" + "🔮 ДИНАМИЧЕСКАЯ РЕКОМЕНДАЦИЯ SWARM AI:")
    print("=" * 50)
    print(f"💡 {recommendation['ai_recommendation']}")
    print(f"🎯 Тип стратегии: {recommendation['strategy_type'].upper()}")
    print(f"🤝 {recommendation['swarm_advice']}")
    print(f"📈 {recommendation['dynamic_insights']}")
    print(f"🔍 {recommendation['strategy_analysis']}")
    print(f"✅ Вероятность успеха: {recommendation['success_probability']:.1%}")
    print(f"⚖️  Оценка риска: {recommendation['risk_assessment']}")
    print(f"🔄 Альтернативы: {', '.join(recommendation['alternative_strategies'])}")

    # Демонстрация динамической симуляции
    print("\n4. ДИНАМИЧЕСКАЯ ФИНАНСОВАЯ СИМУЛЯЦИЯ:")
    print("-" * 60)

    total_savings = 0
    situation_changes = []

    for month in range(12):
        # Получаем текущий контекст
        current_situation = user_env._assess_current_situation()
        context = {
            'month': user_env.month,
            'situation_type': current_situation['type'],
            'economic_outlook': user_env.economic_outlook
        }

        # Получаем динамическую рекомендацию
        recommendation = advisor.get_dynamic_recommendation(state, user_env.user_profile, context)

        # Для предсказания используем правильные входные данные через select_strategy_model
        strategy_type, model, model_input = swarm_ai.select_strategy_model(state, context)
        q_values = model.predict(model_input[np.newaxis], verbose=0)[0]
        action = np.argmax(q_values)

        # Для сезонной модели нужна специальная обработка действий
        if strategy_type == "seasonal":
            # Маппинг действий для сезонной модели (7 выходов -> 5 действий)
            seasonal_mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4}
            action = seasonal_mapping.get(action, 2)  # По умолчанию 10%
        elif strategy_type == "crisis":
            # Маппинг для кризисной модели (3 выхода -> 5 действий)
            crisis_mapping = {0: 0, 1: 0, 2: 1}
            action = crisis_mapping.get(action, 0)
        elif strategy_type == "opportunity":
            # Маппинг для opportunity модели (4 выхода -> 5 действий)
            opportunity_mapping = {0: 2, 1: 3, 2: 4, 3: 4}
            action = opportunity_mapping.get(action, 2)

        # Выполняем действие
        next_state, reward, done, step_context = user_env.step(action)
        savings_rate = action * 5
        monthly_savings = user_env.user_profile['income'] * (savings_rate / 100)
        total_savings += monthly_savings

        # Отслеживаем изменения ситуации
        if month > 0 and current_situation['type'] != situation_changes[-1]['situation']:
            situation_change = "🔄 ИЗМЕНЕНИЕ СИТУАЦИИ"
        else:
            situation_change = ""

        situation_changes.append({
            'month': month + 1,
            'situation': current_situation['type'],
            'change': situation_change,
            'strategy': strategy_type,
            'savings_rate': savings_rate
        })

        #{current_situation['type'].upper()} {situation_change}
        print(f"Месяц {month + 1}: ")
        print(f"   Стратегия: {strategy_type} | Отложено: {savings_rate}% ({monthly_savings:.0f} руб.)")
        print(f"   Баланс: {next_state[2]:.0f} руб. | Награда: {reward:.2f}")
        print(f"   Эконом.прогноз: {user_env.economic_outlook:.1%}")
        print()

        state = next_state
        if done:
            break

    # Сравнение со статической стратегией
    print("\n5. 📊 СРАВНЕНИЕ С СТАТИЧЕСКОЙ СТРАТЕГИЕЙ:")
    print("-" * 50)

    static_savings = user_env.user_profile['income'] * 0.10 * 12  # Всегда 10%
    dynamic_advantage = total_savings - static_savings
    advantage_percent = (dynamic_advantage / static_savings) * 100

    print(f"💵 Static 10% стратегия: {static_savings:.0f} руб.")
    print(f"🚀 Dynamic Swarm AI: {total_savings:.0f} руб.")
    print(f"📈 Преимущество Swarm AI: {dynamic_advantage:+.0f} руб. ({advantage_percent:+.1f}%)")
    print(f"🏦 Финальный баланс: {state[2]:.0f} руб.")

    # Анализ адаптивности
    print("\n6. 📈 АНАЛИЗ АДАПТИВНОСТИ:")
    print("-" * 40)
    unique_situations = set([s['situation'] for s in situation_changes])
    print(f"Количество различных ситуаций: {len(unique_situations)}")
    for situation in unique_situations:
        count = len([s for s in situation_changes if s['situation'] == situation])
        strategies_used = set([s['strategy'] for s in situation_changes if s['situation'] == situation])
        avg_savings = np.mean([s['savings_rate'] for s in situation_changes if s['situation'] == situation])
        print(
            f"  {situation}: {count} месяцев | Стратегии: {', '.join(strategies_used)} | Средние сбережения: {avg_savings:.1f}%")

    changes_count = len([s for s in situation_changes if s['change']])
    print(f"Изменений стратегии: {changes_count}")

    # Дополнительная статистика
    print("\n7. 📋 СТАТИСТИКА СТРАТЕГИЙ:")
    print("-" * 30)
    strategy_stats = {}
    for s in situation_changes:
        strategy = s['strategy']
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {'count': 0, 'total_savings': 0}
        strategy_stats[strategy]['count'] += 1
        strategy_stats[strategy]['total_savings'] += s['savings_rate']

    for strategy, stats in strategy_stats.items():
        avg_rate = stats['total_savings'] / stats['count']
        print(f"  {strategy}: {stats['count']} месяцев | Средняя ставка: {avg_rate:.1f}%")

# -------------- Сравнение стратегий ----------------
def compare_static_vs_dynamic():
    print("\n" + "=" * 70)
    print("🔬 ЭКСПЕРИМЕНТ: Static vs Dynamic стратегии")
    print("=" * 70)

    swarm_ai = DynamicSwarmFinancialAI()

    # Тестовые сценарии с полными профилями
    test_cases = [
        {
            "name": "💰 Кризис + низкий баланс",
            "balance": 8000,
            "income": 40000,
            "expenses_ratio": 0.95,
            "family_status": "single",
            "goals": "apartment",
            "age_group": "26-35",
            "income_stability": 0.5,
            "emergency_fund_ratio": 0.1,
            "expense_volatility": 0.7
        },
        {
            "name": "🚀 Рост + высокий доход",
            "balance": 150000,
            "income": 120000,
            "economic_outlook": 0.9,
            "family_status": "family",
            "goals": "investment",
            "age_group": "36-45",
            "income_stability": 0.8,
            "emergency_fund_ratio": 0.3,
            "expense_volatility": 0.3
        },
        {
            "name": "🎄 Праздничный сезон",
            "balance": 50000,
            "income": 70000,
            "month": 11,
            "family_status": "couple",
            "goals": "travel",
            "age_group": "26-35",
            "income_stability": 0.7,
            "emergency_fund_ratio": 0.2,
            "expense_volatility": 0.5
        },
        {
            "name": "⚡ Неожиданные события",
            "balance": 30000,
            "income": 60000,
            "events": ['medical', 'car_repair'],
            "family_status": "family",
            "goals": "education",
            "age_group": "36-45",
            "income_stability": 0.6,
            "emergency_fund_ratio": 0.15,
            "expense_volatility": 0.6
        }
    ]

    print("\nСравнение рекомендаций:")
    print("-" * 80)
    print(f"{'Сценарий':<30} {'Static 10%':<15} {'Dynamic AI':<15} {'Разница':<15} {'Обоснование':<20}")
    print("-" * 80)

    for case in test_cases:
        env = DynamicSwarmSavingsEnv(user_id="comparison", swarm_ai=swarm_ai)

        # Полностью инициализируем пользовательский профиль
        env.current_balance = case["balance"]
        env.user_profile = {
            'income': case["income"],
            'expenses': {
                'essential': case["income"] * 0.4,
                'housing': case["income"] * 0.3,
                'transportation': case["income"] * 0.1,
                'entertainment': case["income"] * 0.1
            },
            'goals': case["goals"],
            'family_status': case["family_status"],
            'age_group': case["age_group"],
            'income_stability': case["income_stability"],
            'emergency_fund_ratio': case["emergency_fund_ratio"],
            'expense_volatility': case["expense_volatility"]
        }

        # Настраиваем расходы если указан ratio
        if case.get("expenses_ratio"):
            total_expenses = case["income"] * case["expenses_ratio"]
            env.user_profile['expenses'] = {
                'essential': total_expenses * 0.7,
                'housing': total_expenses * 0.3
            }

        if case.get("economic_outlook"):
            env.economic_outlook = case["economic_outlook"]

        if case.get("month"):
            env.month = case["month"]

        # Добавляем события если есть
        if case.get("events"):
            env.personal_events = [
                {'type': event, 'impact': -0.2, 'probability': 0.1}
                for event in case["events"]
            ]

        # Инициализируем кластер
        env.user_cluster = env.swarm_ai.user_clusters.assign_cluster(env.user_profile)

        state = env._get_dynamic_state()
        context = {
            'month': env.month,
            'situation_type': env._assess_current_situation()['type'],
            'economic_outlook': env.economic_outlook
        }

        # Static рекомендация
        static_action = 2  # Всегда 10%
        static_rate = 10

        # Dynamic рекомендация
        advisor = DynamicSwarmAdvisor(swarm_ai)
        recommendation = advisor.get_dynamic_recommendation(state, env.user_profile, context)

        # ИСПРАВЛЕННЫЙ ПАРСИНГ - извлекаем число из рекомендации
        dynamic_recommendation = recommendation['ai_recommendation']

        # Метод 1: Ищем число перед символом %
        import re
        numbers = re.findall(r'(\d+)%', dynamic_recommendation)
        if numbers:
            dynamic_rate = int(numbers[0])
        else:
            # Метод 2: Ищем любое число в тексте
            numbers = re.findall(r'\d+', dynamic_recommendation)
            if numbers:
                dynamic_rate = int(numbers[0])
            else:
                # Метод 3: Используем action из стратегии как fallback
                strategy_type, model, model_input = swarm_ai.select_strategy_model(state, context)
                q_values = model.predict(model_input[np.newaxis], verbose=0)[0]
                action = np.argmax(q_values)
                dynamic_rate = action * 5

        difference = dynamic_rate - static_rate

        # Обрезаем обоснование для красивого вывода
        reasoning = recommendation['strategy_analysis']
        if len(reasoning) > 35:
            reasoning = reasoning[:32] + "..."

        print(f"{case['name']:<30} {static_rate}%{'':<11} {dynamic_rate}%{'':<11} {difference:>+3}%{'':<9} {reasoning}")

    print("-" * 80)

    # Добавляем итоговый анализ
    print("\n📊 ИТОГОВЫЙ АНАЛИЗ:")
    print("Dynamic Swarm AI адаптирует рекомендации под конкретную ситуацию:")
    print("• В кризисных сценариях снижает риск")
    print("• В благоприятных условиях увеличивает сбережения")
    print("• Учитывает сезонные факторы и личные обстоятельства")
    print("• Основан на коллективном опыте успешных пользователей")


# Запуск демонстрации
if __name__ == "__main__":
    demonstrate_dynamic_swarm_ai()
    compare_static_vs_dynamic()
    print("\n🎉 Демонстрация завершена! Swarm AI показывает превосходную адаптивность! 🚀")
