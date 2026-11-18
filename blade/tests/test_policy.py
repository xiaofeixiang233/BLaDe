import json
import math
import unittest
from blade import policy


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.menu=['forward','pickup','left']
        self.public={'environment':'babyai','mission':'pick up the key',
          'observation':{'long_term_context':'key directly ahead','short_term_context':'forward failed'},
          'native_instruction':'native rules','action_descriptions':{a:a for a in self.menu},
          'reward':999,'hidden_answer':'pickup'}
        self.history=[{'action':'forward','after':{'mission':'pick up the key','observation':'blocked'+str(i)},'reward':999} for i in range(8)]
        self.clock=policy.goal_time(100,8,8,4)

    def test_baselines_use_only_pi0_from_first_action(self):
        for kind in ['jev','d3']:
            for step in [0,8,99]:self.assertEqual(policy.roles_for(kind+'-flat-h0',step),('pi0',))
            self.assertEqual(policy.roles_for(kind+'-w2',0),('ph',))
            self.assertEqual(policy.roles_for(kind+'-w05',8),('pi0','ph','q'))

    def test_independent_inputs_no_predictions_rewards_or_hidden_state(self):
        for role in ['pi0','ph','q']:
            p=policy.payload(role,self.public,self.menu,self.history,{'goal':'hold key'},self.clock)
            s=p['state']
            self.assertEqual('previous_interactions' in s,role=='ph')
            self.assertEqual('desired_state' in s,role=='q')
            self.assertNotIn('reward',json.dumps(s));self.assertNotIn('hidden_answer',json.dumps(s))
            self.assertEqual(list(p['questions']['action']['criteria']),self.menu)
            self.assertEqual(s['observation'],self.public['observation'])
        h=policy.payload('ph',self.public,self.menu,self.history,None,self.clock)['state']['previous_interactions']
        self.assertEqual(len(h),4);self.assertEqual(h[0]['observation'],'blocked4')

    def test_h0_does_not_even_inspect_history_or_goal(self):
        policy.payload('pi0',self.public,self.menu,object(),object(),self.clock)

    def test_goal_scheduling_is_fixed_not_completion_dependent(self):
        self.assertEqual(policy.goal_slots(100,4),[8,31,54,77])
        self.assertEqual(policy.goal_slots(2000,8),[8,257,506,755,1004,1253,1502,1751])
        self.assertEqual(policy.goal_slots(100000,8)[1],12507)
        self.assertEqual(policy.goal_slots(8,4),[])
        self.assertEqual(policy.goal_time(100,30,8,4)['remaining_goal_horizon'],1)

    def test_effective_decision_cap_preserves_native_horizon(self):
        cfg={'decision_caps':{'crafter':2000,'nle':2000}}
        for name,horizon,expected in [('nle',100000,2000),('crafter',2000,2000),
                ('babyai',64,64),('minihack',100,100),('babaisai',100,100)]:
            case={'environment':name,'horizon':horizon}
            self.assertEqual(policy.effective_decision_cap(cfg,case),expected)
            self.assertEqual(case['horizon'],horizon)
        self.assertEqual(policy.effective_decision_cap(cfg,{'environment':'nle','horizon':100}),100)
        for invalid in [0,-1,True,1.5]:
            with self.assertRaises(ValueError):
                policy.effective_decision_cap({'decision_caps':{'nle':invalid}},
                                             {'environment':'nle','horizon':100000})
        clock=policy.goal_time(2000,1999,1751,8)
        self.assertEqual(clock,{'actions_taken':1999,'remaining_episode_actions':1,
                                'remaining_goal_horizon':1})
        with self.assertRaises(ValueError):policy.goal_time(2000,2000,1751,8)

    def test_goal_request_single_state_not_action_or_oracle(self):
        cfg={'model':'gpt-6-sol','reasoning_effort':'medium','max_output_tokens':4096}
        p=policy.goal_payload(self.public,self.menu,self.history,self.clock,cfg)
        self.assertEqual(p['text']['format']['schema']['required'],['goal'])
        self.assertFalse(p['text']['format']['schema']['additionalProperties'])
        data=json.loads(p['input']);self.assertNotIn('reward',p['input']);self.assertNotIn('hidden_answer',p['input'])
        self.assertEqual(len(data['previous_interactions']),4)
        self.assertNotIn('predicted_state',data)

    def test_weights_apply_to_whole_ratio(self):
        p0={'forward':.6,'pickup':.3,'left':.1};ph={'forward':.5,'pickup':.3,'left':.2};q={'forward':.2,'pickup':.5,'left':.3}
        h=policy.smooth_distribution(self.menu,ph);p=policy.smooth_distribution(self.menu,q);b=policy.smooth_distribution(self.menu,p0)
        for method,w in [('jev-w2',2),('d3-w05',.5)]:
            scores=policy.log_action_scores(method,self.menu,ph=ph,q=q,p0=p0)
            for a in self.menu:self.assertAlmostEqual(scores[a],math.log(h[a])+w*(math.log(p[a])-math.log(b[a])))
            distribution=policy.combine_distributions(method,self.menu,ph=ph,q=q,p0=p0)
            self.assertAlmostEqual(sum(distribution.values()),1)

    def test_no_state_legality_mask_even_when_action_failed_before(self):
        d={'forward':1.,'pickup':0.,'left':0.}
        self.assertEqual(policy.choose_action(self.menu,d),'forward')
        result=policy.combine_distributions('jev-w2',self.menu,ph=d,q=d,p0=d)
        self.assertEqual(set(result),set(self.menu))
        self.assertTrue(all(math.isfinite(x) and x>0 for x in result.values()))

    def test_ties_use_native_order(self):
        self.assertEqual(policy.choose_action(['b','a'],{'a':.5,'b':.5}),'b')


if __name__=='__main__':unittest.main()
